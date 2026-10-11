"""Attribute remaining scene errors without mining development pixels or tuning thresholds."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_false_positives import candidate_key, classify_errors
from scripts.audit_blue_final_errors import missed_stages
from scripts.evaluate_red_blue_development import LABELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_real_blue_clutter import OUTPUT as DECISION

OUTPUT = Path('research/balloon-red-blue-20261008/real-blue-clutter-errors-20261011.json')


def attribute(result, baseline, parent):
    metrics = checked_metrics(result)
    fp, fn, covered, total = Counter(), Counter(), Counter(), Counter()
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
    for color in LABELS:
        if (sum(v for k, v in fp.items() if k[0] == color) != metrics[color]['fp']
                or sum(v for k, v in fn.items() if k[0] == color) != metrics[color]['fn']):
            raise ValueError('Error totals do not reproduce scene metrics')
    return dict(metrics=metrics,
        false_positive_counts={'/'.join(k): v for k, v in sorted(fp.items())},
        false_negative_counts={'/'.join(k): v for k, v in sorted(fn.items())},
        geometric_target_coverage={c: dict(covered=covered[c], total=total[c],
            fraction=covered[c] / total[c] if total[c] else None) for c in LABELS})


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    decision = read_json(DECISION)
    panels = {}
    for panel, identity in decision['panels'].items():
        if sha256(identity['path']) != identity['sha256']:
            raise ValueError('Audited scene results changed')
        source = read_json(identity['path'])
        variants = {}
        for name, result in source['results'].items():
            model, search = name.split('/')
            variants[name] = attribute(result, source['results'][f'{model}/baseline']['frames'],
                                      source['results'][f'parent/{search}']['frames'])
        panels[panel] = dict(source_sha256=identity['sha256'], results=variants)
    write_json(OUTPUT, dict(panels=panels, decision_sha256=sha256(DECISION),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/audit_balloon_false_positives.py', 'scripts/audit_blue_final_errors.py',
            'scripts/evaluate_red_blue_development.py', 'scripts/summarize_public_balloon_trial.py')},
        scope='Saved development geometry only. Zero annotated overlap does not prove background. Per-target proposal coverage is an optimistic bound, not attainable recall. No image mining, threshold selection, or qualification.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print({p: r['results']['admit48/coarse47'] for p, r in panels.items()})


if __name__ == '__main__':
    audit()
