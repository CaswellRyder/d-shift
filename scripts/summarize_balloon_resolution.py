"""Recompute native-detail proposal ceilings and reject unsupported promotion."""
import argparse
import math
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.evaluate_balloon_edge_search import coverage


def bounded_supplement(base, high):
    """Audit only: preserve baseline and allow at most two new blue crops."""
    found = list(base)
    for proposal in high:
        if len(found) >= min(12, len(base)+2):
            break
        if proposal['color_group'] == 1 and not any(iou(proposal['box'], r['box']) > .65 for r in found):
            found.append(proposal)
    return found


def summarize(path):
    doc = read_json(path)
    if (doc['test_evaluated'] is not False or doc['classifier_evaluated'] is not False
            or doc['deployment_approved'] is not False or doc['pi_tested'] is not False):
        raise ValueError('Unexpected audit scope')
    for source, digest in doc['inputs'].items():
        if sha256(source) != digest:
            raise ValueError('Source identity changed')
    totals = {label: dict(targets=0, baseline=0, high=0, union=0, bounded=0)
              for label in ('red_balloon', 'blue_balloon')}
    times, sources = [], set()
    for panel in doc['panels'].values():
        for frame in panel['frames']:
            if frame['source'] in sources:
                raise ValueError('Duplicate frame source')
            sources.add(frame['source'])
            base, high = frame['baseline_proposals'], frame['high_proposals']
            if len(base) > 12 or len(high) > 12:
                raise ValueError('Crop budget changed')
            for label in totals:
                totals[label]['targets'] += sum(t['label'] == label for t in frame['truth'])
            for variant, candidates in (('baseline', base), ('high', high), ('union', base+high),
                                        ('bounded', bounded_supplement(base, high))):
                observed = coverage(frame['truth'], candidates)
                if variant in ('baseline', 'high') and observed != frame[variant]:
                    raise ValueError('Stored counts differ from one-to-one matching')
                for label in totals:
                    totals[label][variant] += observed[label]['localized']
            if frame['native_detail_used']:
                pair = (frame['baseline_host_search_ms'], frame['high_host_search_ms'])
                if not all(math.isfinite(v) and v > 0 for v in pair):
                    raise ValueError('Invalid host timing')
                times.append(pair)
    for counts in totals.values():
        counts['required_for_94_percent'] = math.ceil(counts['targets']*.94)
        counts['baseline_proposal_recall_ceiling'] = counts['baseline']/counts['targets']
        counts['high_proposal_recall_ceiling'] = counts['high']/counts['targets']
    return dict(source_report=str(path), source_sha256=sha256(path),
        summary_script_sha256=sha256(__file__), inputs=doc['inputs'],
        totals=totals, frames=len(sources), native_detail_frames=len(times),
        host_mean_search_ms={key: sum(t[i] for t in times)/len(times)
                             for i, key in enumerate(('baseline','high'))} if times else {},
        decision='Do not advance higher-resolution MSER to Pi or development testing: only one extra blue target, one fewer red target, and much more search work.',
        next_direction='Classifier-only retraining cannot recover missing candidate boxes. Investigate a genuinely different localization mechanism using admitted training annotations.',
        caveats=['One-to-one TRAIN proposal coverage, not detector accuracy or independent qualification.',
                 'Union is an unlimited proposal oracle; bounded preserves baseline and adds at most two blue proposals within twelve total.',
                 'Host timings are single-pass diagnostics, not a controlled Pi speed estimate.',
                 'Ceilings assume boxes are unchanged by downstream classification/selection; learned localization can change them.'],
        test_evaluated=False, deployment_approved=False, pi_tested=False, qualification_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, summarize(args.input))
