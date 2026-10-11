"""Saved-result geometric error attribution, never development-image training mining."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_balloon_false_positives import classify_errors, candidate_key
from scripts.evaluate_red_blue_development import LABELS, detection_counts
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics


def missed_stages(frame):
    truth, detections, matched = frame['truth'], frame['detections'], set()
    for row in sorted(detections, key=lambda d: -d['score']):
        if not row['accepted'] or row['label'] not in LABELS:
            continue
        overlap, index = max(((iou(t['box'], row['box']), i) for i, t in enumerate(truth)
                              if i not in matched and t['label'] == row['label']), default=(0., -1))
        if overlap >= .5:
            matched.add(index)
    coverage, misses = Counter(), Counter()
    for i, target in enumerate(truth):
        label = target['label']
        localized = [r for r in detections if iou(r['box'], target['box']) >= .5]
        coverage[label] += bool(localized)
        if i in matched:
            continue
        correct = [r for r in localized if r['label'] == label]
        raw = [r for r in correct if r.get('raw_accepted', r['accepted'])]
        if not localized:
            reason = 'no_geometric_proposal'
        elif not correct:
            reason = 'localized_but_wrong_label'
        elif not raw:
            reason = 'correct_label_below_threshold'
        elif not any(r['accepted'] for r in raw):
            reason = 'suppressed_correct_detection'
        else:
            reason = 'one_to_one_matching_competition'
        misses[(label, reason)] += 1
    expected = detection_counts(truth, detections)
    for label in LABELS:
        if sum(v for (c, _), v in misses.items() if c == label) != expected[label]['fn']:
            raise ValueError('Miss attribution disagrees with canonical matching')
    return coverage, misses


def audit():
    output = Path('research/balloon-red-blue-20261008/blue-final-fit-errors-20261011.json')
    if output.exists():
        raise FileExistsError(output)
    decision_path = Path('research/balloon-red-blue-20261008/blue-final-fit-decision-20261011.json')
    decision = read_json(decision_path)
    panels = {}
    for panel in PANELS:
        identity = decision['panels'][panel]
        if sha256(identity['path']) != identity['sha256']:
            raise ValueError('Audited scene results changed')
        source = read_json(identity['path'])
        variants = {}
        for name, result in source['results'].items():
            model, search = name.split('/')
            baseline = source['results'][f'{model}/baseline']['frames']
            parent = source['results'][f'parent/{search}']['frames']
            fp, fn, covered, total = Counter(), Counter(), Counter(), Counter()
            m = checked_metrics(result)
            for frame, base, reference in zip(result['frames'], baseline, parent, strict=True):
                for other in (base, reference):
                    if any(frame[k] != other[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                        raise ValueError('Frame identity changed')
                if [candidate_key(r) for r in frame['detections']] != [candidate_key(r) for r in reference['detections']]:
                    raise ValueError('Classifier changed proposal geometry or ordering')
                fp.update((e['label'], e['origin'], e['reason']) for e in classify_errors(frame, base))
                coverage, misses = missed_stages(frame)
                fn.update(misses)
                covered.update(coverage)
                total.update(t['label'] for t in frame['truth'])
            variants[name] = dict(metrics=m,
                false_positive_counts={'/'.join(k): v for k, v in sorted(fp.items())},
                false_negative_counts={'/'.join(k): v for k, v in sorted(fn.items())},
                geometric_target_coverage={c: dict(covered=covered[c], total=total[c],
                    fraction=covered[c] / total[c] if total[c] else None) for c in LABELS})
        panels[panel] = dict(source_sha256=identity['sha256'], results=variants)
    write_json(output, dict(panels=panels, decision_sha256=sha256(decision_path),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/audit_balloon_false_positives.py', 'scripts/evaluate_red_blue_development.py')},
        scope='Geometric attribution only. Zero annotated overlap is not visual proof of background. Per-target geometry coverage is an optimistic upper bound, not jointly attainable recall. No pixels mined, thresholds selected or labels changed.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print({p: {n: v for n, v in r['results'].items() if n in ('parent/coarse47', 'admit47/coarse47', 'admit48/coarse47')}
           for p, r in panels.items()})


if __name__ == '__main__':
    audit()
