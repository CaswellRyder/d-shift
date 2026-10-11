"""Saved TRAIN-search false-alarm and target-retention audit; no tuning."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.compare_current_blue_search import OUTPUT as TRIAL, RAW, MODES
from scripts.audit_balloon_false_positives import classify_errors
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.evaluate_red_blue_development import LABELS

OUTPUT = Path('research/balloon-red-blue-20261008/current-blue-search-errors-20261011.json')


def matched_truth(frame):
    matched = set()
    for row in sorted(frame['detections'], key=lambda d: -d['score']):
        if not row['accepted'] or row['label'] not in LABELS:
            continue
        overlap, index = max(((iou(t['box'], row['box']), i) for i, t in enumerate(frame['truth'])
                              if i not in matched and t['label'] == row['label']), default=(0., -1))
        if overlap >= .5:
            matched.add(index)
    return matched


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial, raw = read_json(TRIAL), read_json(RAW)
    if (trial['raw']['sha256'] != sha256(RAW)
            or trial['script_sha256'] != sha256('scripts/compare_current_blue_search.py')
            or any(sha256(p) != h for p, h in trial['helpers_sha256'].items())):
        raise ValueError('TRAIN search comparison changed')
    results = {}
    for mode in MODES[1:]:
        result = raw['results'][mode]
        if checked_metrics(result) != trial['results'][mode]['metrics']:
            raise ValueError('Metrics changed')
        counts, errors, gained, lost = Counter(), [], [], []
        for row, base in zip(result['frames'], raw['results']['mser']['frames'], strict=True):
            if any(row[k] != base[k] for k in ('source', 'frame_sha256', 'truth')):
                raise ValueError('Comparison frame identity changed')
            for error in classify_errors(row, base):
                counts[(error['label'], error['origin'], error['reason'])] += 1
                a, b, c, d = error['box']
                errors.append(dict(error, source=row['source'], touches_image_edge=a <= 0 or b <= 0 or c >= 320 or d >= 240))
            old, new = matched_truth(base), matched_truth(row)
            gained.extend(dict(source=row['source'], target=row['truth'][i]) for i in sorted(new - old))
            lost.extend(dict(source=row['source'], target=row['truth'][i]) for i in sorted(old - new))
        results[mode] = dict(error_counts={'/'.join(k): v for k, v in sorted(counts.items())},
            errors=errors, gained_targets=gained, lost_targets=lost)
    write_json(OUTPUT, dict(results=results, trial_sha256=sha256(TRIAL), raw_sha256=sha256(RAW),
        script_sha256=sha256(__file__),
        scope='Known complete-reviewed FIT scenes only. Geometric association is not visual relabeling. No thresholds changed and no runtime filter selected.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print({mode: dict(error_counts=r['error_counts'], gained=len(r['gained_targets']), lost=len(r['lost_targets'])) for mode, r in results.items()})


if __name__ == '__main__':
    audit()
