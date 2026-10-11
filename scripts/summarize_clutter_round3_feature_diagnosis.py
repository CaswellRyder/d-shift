"""Keep the large TRAIN replay artifact ignored and commit its recounted summary."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_clutter_round3_feature import OUTPUT as AUDIT
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.summarize_public_balloon_trial import checked_metrics

RAW = Path('runs/clutter-round3-feature-diagnosis-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/clutter-round3-feature-diagnosis-summary-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    raw, audit = read_json(RAW), read_json(AUDIT)
    if (raw['audit_sha256'] != sha256(AUDIT)
            or raw['script_sha256'] != sha256('scripts/diagnose_clutter_round3_feature.py')
            or any(sha256(p) != h for p, h in raw['helpers_sha256'].items())
            or any(raw[k] is not False for k in ('development_evaluated', 'test_evaluated', 'pi_tested',
                                                 'deployment_approved', 'qualification_achieved'))
            or set(raw['results']) != set(audit['results'])):
        raise ValueError('Diagnostic identity or scope changed')
    baseline = checked_metrics(raw['parent'])
    results = {}
    for name, r in raw['results'].items():
        measured = checked_metrics(r)
        comparison = compare_frames(r['frames'], raw['parent']['frames'])
        if comparison != r['comparison']:
            raise ValueError('Identity-level counts changed')
        results[name] = dict(metrics=measured, comparison=comparison,
            raw_decision_changes=r['raw_decision_changes'], failed_new_backgrounds=r['failed_new_backgrounds'],
            training_gate_pass=audit['results'][name]['training_gate_pass'])
    write_json(OUTPUT, dict(raw=dict(path=str(RAW), sha256=sha256(RAW)),
        audit_sha256=sha256(AUDIT), script_sha256=sha256(__file__), parent_metrics=baseline,
        results=results, decision='No promotion: neither admission seed passed all preregistered TRAIN checks.',
        next_experiment='Test a minimal head correction on the newly learned features, with reference-parent correctness constraints, instead of enlarging the network. Keep matched controls and seeds; do not retune on development.',
        limitations=['This is cached TRAIN score replay, not fresh search execution or independent accuracy.',
            'Preserving every raw proposal can preserve suppressed/localization-error detections; inspect target-level losses separately.',
            'Both admissions retained TRAIN scene targets; seed48 introduced a red false alarm. Three new negatives remain unlearned.',
            'No export, development evaluation, Pi replacement, ESP32 or motor commands in this experiment.'],
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(summary=str(OUTPUT), raw_bytes=RAW.stat().st_size, summary_bytes=OUTPUT.stat().st_size), flush=True)


if __name__ == '__main__':
    summarize()
