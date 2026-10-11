"""Recount full-scene outcomes and record the fixed-duration retraining decision."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_background_convergence import OUTPUT as SCENES, BASELINE, paired_survivors
from scripts.evaluate_current_confident_containment import nonregressing
from scripts.evaluate_red_blue_development import LABELS
from scripts.export_background_convergence import checked_trial
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/background-convergence-decision-20261011.json')


def recount(doc, raw, baseline, names):
    if set(raw) != set(doc['panels']) or set(raw) != set(baseline) or len(raw) != 4:
        raise ValueError('Incomplete scene comparison')
    summaries = {}
    for panel, variants in raw.items():
        if set(variants) != {'parent', *names} or set(doc['panels'][panel]) != set(variants):
            raise ValueError('Incomplete paired panel')
        parent = variants['parent']['frames']
        for a, b in zip(parent, baseline[panel]['frames'], strict=True):
            check_baseline(a, b)
        summaries[panel] = {}
        for name, result in variants.items():
            measured = checked_metrics(result)
            compared = compare_frames(result['frames'], parent)
            gate = nonregressing(variants['parent']['metrics'], measured, compared)
            calls = sum(f['neural_calls'] for f in result['frames'])
            expected = dict(metrics=measured, comparison=compared, nonregression=gate,
                meets_94=all(measured[c][m] >= .94 for c in LABELS for m in ('precision', 'recall')),
                neural_calls=calls)
            if expected != doc['panels'][panel][name]:
                raise ValueError('Scene summaries do not reproduce')
            for frame, original in zip(result['frames'], parent, strict=True):
                if frame['neural_calls'] != original['neural_calls']:
                    raise ValueError('Unequal classification workload')
                for a, b in zip(frame['detections'], original['detections'], strict=True):
                    if any(a[k] != b[k] for k in ('box', 'crop_box')):
                        raise ValueError('Classifier comparison changed proposal geometry')
            summaries[panel][name] = dict(metrics=measured, nonregression=gate,
                lost=len(compared['lost_parent_targets']), gained=len(compared['gained_targets']),
                errors={c: compared['colors'][c]['errors'] for c in LABELS})
    train_survivors = paired_survivors(names, doc['panels']['training'])
    viable = [n for n in train_survivors if all(p[n]['nonregression'] for p in doc['panels'].values())]
    viable = paired_survivors(viable, doc['panels']['training'])
    if train_survivors != doc['eligible_for_development'] or viable != doc['eligible_for_pi_comparison']:
        raise ValueError('Eligibility does not reproduce')
    if (doc['development_evaluated'] is not True or any(doc[k] is not False for k in
            ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Unsupported scope claim')
    return summaries, viable


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    _, names = checked_trial()
    doc, reference = read_json(SCENES), read_json(BASELINE)
    if (sha256(doc['raw']['path']) != doc['raw']['sha256']
            or sha256(BASELINE) != doc['baseline_sha256']
            or sha256(reference['raw']['path']) != reference['raw']['sha256']
            or sha256('scripts/evaluate_background_convergence.py') != doc['script_sha256']
            or any(sha256(p) != h for p, h in {**doc['helpers_sha256'], **doc['inputs']}.items())):
        raise ValueError('Scene evaluation evidence changed')
    for export in doc['exports'].values():
        path = Path(export['path'])
        if (sha256(path) != export['sha256']
                or sha256(path.with_suffix('.json')) != export['metadata_sha256']):
            raise ValueError('Export changed')
    summaries, viable = recount(doc, read_json(doc['raw']['path']), read_json(reference['raw']['path']), names)
    if viable:
        raise ValueError('This rejection receipt requires no surviving paired arm')
    result = dict(scene_receipt_sha256=sha256(SCENES), script_sha256=sha256(__file__), summaries=summaries,
        decision='Do not replace the current classifier or stage these checkpoints for Pi promotion.',
        evidence='All four400epoch checkpoints pass crop retention and64TRAIN scene nonregression. Both seeds of both losses lose targets on each of the two larger repeated development panels; the12frame panel is unchanged. Training fit did not transfer reliably.',
        next_direction='Use TRAIN-only acquisition or augmentation to address missing real blue appearance coverage; preserve protected development and reserved test. Inspect stage-specific errors before selecting another change. More epochs alone are not supported by this result.',
        scope='Audited fixed-checkpoint research comparison, not independent accuracy or live Pi evidence. Related development panels must not be pooled as independent samples.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(OUTPUT, result)
    print(result['decision'], flush=True)
    for panel, variants in summaries.items():
        print(panel, {n: r['errors']['blue_balloon'] for n, r in variants.items()}, flush=True)


if __name__ == '__main__':
    audit()
