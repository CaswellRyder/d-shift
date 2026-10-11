"""Attribute frozen scene errors; ground-truth oracles are NOT runtime policies."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_pipeline_errors import analyze_frame, maximum_matching
from scripts.evaluate_dense_context_trial import OUTPUT as DEVELOPMENT, RAW
from scripts.evaluate_red_blue_development import LABELS
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_dense_context_trial import OUTPUT as DECISION
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/dense-context-trial-errors-20261011.json')
PREVIOUS = Path('runs/blue-photo-convergence-development-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc, decision, raw, previous = map(read_json, (DEVELOPMENT, DECISION, RAW, PREVIOUS))
    if (doc['raw']['sha256'] != sha256(RAW)
            or decision['development_receipt_sha256'] != sha256(DEVELOPMENT)
            or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Require unchanged unapproved development evidence')
    panels = {}
    for panel, variants in raw.items():
        results = {}
        for name, result in variants.items():
            if checked_metrics(result) != doc['panels'][panel]['results'][name]:
                raise ValueError('Metrics do not reproduce')
            # Current compact controls have precisely the preceding experiment's
            # admitted data and schedule. Verify every prediction, not just counts.
            if name.startswith('control'):
                prior = previous[panel][name.replace('control', 'admit', 1)]['frames']
                for row, old in zip(result['frames'], prior, strict=True):
                    check_baseline(row, old)
            frames = [analyze_frame(f) for f in result['frames']]
            colors = {}
            for label in LABELS:
                count = sum(t['label'] == label for f in result['frames'] for t in f['truth'])
                covered = sum(len(maximum_matching(
                    [t for t in f['truth'] if t['label'] == label], f['detections'],
                    lambda t, d: True)) for f in result['frames'])
                colors[label] = dict(
                    targets=count, geometric_oracle_covered=covered,
                    geometric_oracle_recall=covered / count if count else None,
                    errors={kind: dict(Counter(e['reason'] for f in frames for e in f[kind]
                                               if e['label'] == label))
                            for kind in ('false_positives', 'false_negatives')})
            results[name] = dict(metrics=result['metrics'], colors=colors, frames=frames)
        panels[panel] = results
    write_json(OUTPUT, dict(
        panels=panels, development_sha256=sha256(DEVELOPMENT), decision_sha256=sha256(DECISION),
        raw_sha256=sha256(RAW), previous_raw_sha256=sha256(PREVIOUS),
        compact_control_predictions_reproduce_previous_admitted=True,
        script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in (
            'scripts/audit_balloon_pipeline_errors.py', 'scripts/summarize_public_balloon_trial.py',
            'scripts/summarize_dense_background_trial.py')},
        scope='Repeated related development. Per-color geometric oracle uses truth and perfect classification; not achievable accuracy or a deployment policy.',
        test_evaluated=False, pi_tested=False, training_performed=False,
        deployment_approved=False, qualification_achieved=False))
    for panel, variants in panels.items():
        print(panel, {k: v['colors'] for k, v in variants.items() if k.endswith('coarse47')})


if __name__ == '__main__':
    audit()
