"""Verify fixed80 continuation and retain the original TRAIN eligibility gate."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_protected_clutter import checked_run as parent_run
from scripts.balloon_positive_protection import retention, feasible
from scripts.continue_balloon_protected_clutter import PROTOCOL, LOSS_PROTOCOL


def checked_run(arm, seed):
    parent, pp, pr, old = parent_run(arm, seed)
    run = Path(f'runs/balloon-protected-convergence-{arm}-seed{seed}-20261011')
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete'
            or p['arm'] != arm or p['seed'] != seed or p['parent'] != str(parent)
            or p['parent_model_sha256'] != pr['model_sha256']
            or p['parent_provenance_sha256'] != pr['provenance_sha256']
            or p['parent_predictions_sha256'] != pr['predictions_sha256']
            or p['input_tensor_sha256'] != pp['input_tensor_sha256']
            or p['parameters'] != 7763 or p['epochs'] != 80 or p['initial_epoch'] != 24
            or p['protocol_sha256'] != sha256(PROTOCOL) or p['loss_protocol_sha256'] != sha256(LOSS_PROTOCOL)
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())
            or p['original_parent_targets_preserved'] is not True or p['fixed_final_epoch'] is not True):
        raise ValueError('Continuation identity changed')
    for receipt, steps in ((p['optimizer_restored'], 2304), (r['optimizer_final'], 7680)):
        if (receipt['iterations'] != steps or receipt['finite_slots'] is not True
                or receipt['nonzero_slots'] is not True
                or not np.isclose(receipt['learning_rate'], .0001, rtol=0, atol=1e-10)
                or receipt['variables'] != 2 + 2 * receipt['trainable_tensors']):
            raise ValueError('Optimizer continuation proof missing')
    with np.load(run / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ('before', 'labels', 'ordinary', 'probe_before', 'unprotected'):
        if not np.array_equal(arrays[key], old[key]):
            raise ValueError('Original parent targets or labels changed')
    if (r['retention'] != retention(arrays['before'], arrays['after'], arrays['labels'], arrays['ordinary'])
            or arrays['probe_after'].shape != (22, 3) or not np.isfinite(arrays['probe_after']).all()
            or r['reviewed_background_correct'] != int((arrays['probe_after'].argmax(1) == 0).sum())
            or r['probe_identity'] != pr['probe_identity']
            or r['training_gate_pass'] != feasible(r, read_json(LOSS_PROTOCOL))):
        raise ValueError('Continuation metrics differ from predictions')
    return run, p, r, arrays


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    trials = {}
    for arm in ('control', 'admit'):
        for seed in (47, 48):
            run, p, r, arrays = checked_run(arm, seed)
            trials[f'{arm}{seed}'] = dict(path=str(run), model_sha256=r['model_sha256'],
                predictions_sha256=r['predictions_sha256'], provenance_sha256=r['provenance_sha256'],
                retention=r['retention'], reviewed_background_correct=r['reviewed_background_correct'],
                reviewed_background_by_view={view: int((arrays['probe_after'][start:start + 11].argmax(1) == 0).sum())
                    for view, start in (('original', 0), ('blue_surface', 11))},
                training_gate_pass=r['training_gate_pass'], optimizer_restored=p['optimizer_restored'],
                optimizer_final=r['optimizer_final'])
    eligible = all(trials[f'admit{s}']['training_gate_pass'] for s in (47, 48))
    write_json(output, dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), loss_protocol_sha256=sha256(LOSS_PROTOCOL), script_sha256=sha256(__file__),
        decision=('Both seeds pass unchanged TRAIN feasibility; development still required.' if eligible else
                  'Do not export or evaluate development: final80 checkpoints fail unchanged TRAIN feasibility.'),
        scope='Matched TRAIN continuation, not new held-out scene or live Pi evidence.',
        test_evaluated=False, development_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible, trials=trials))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
