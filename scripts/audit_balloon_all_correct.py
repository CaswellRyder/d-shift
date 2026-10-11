"""Verify the protection-mask ablation before opening any development evaluation."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_protected_convergence import checked_run as parent_run
from scripts.balloon_positive_protection import retention, feasible
from scripts.train_balloon_all_correct import PROTOCOL, LOSS_PROTOCOL


def checked_run(arm, seed):
    parent, pp, pr, old = parent_run('admit', seed)
    run = Path(f'runs/balloon-all-correct-{arm}-seed{seed}-20261011')
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete'
            or p['arm'] != arm or p['seed'] != seed or p['parent'] != str(parent)
            or p['parent_model_sha256'] != pr['model_sha256']
            or p['parent_provenance_sha256'] != pr['provenance_sha256']
            or p['parent_predictions_sha256'] != pr['predictions_sha256']
            or p['input_tensor_sha256'] != pp['input_tensor_sha256']
            or p['parameters'] != 7763 or p['final_epoch'] != 104 or p['initial_epoch'] != 80
            or p['protocol_sha256'] != sha256(PROTOCOL) or p['loss_protocol_sha256'] != sha256(LOSS_PROTOCOL)
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())
            or p['original_parent_targets_preserved'] is not True or p['fixed_final_epoch'] is not True
            or p['optimizer_slots_preserved_on_recompile'] is not True):
        raise ValueError('Ablation checkpoint or sources changed')
    for receipt, steps in ((p['optimizer_restored'], 7680), (r['optimizer_final'], 9984)):
        if (receipt['iterations'] != steps or receipt['finite_slots'] is not True
                or receipt['nonzero_slots'] is not True
                or not np.isclose(receipt['learning_rate'], .0001, rtol=0, atol=1e-10)
                or receipt['variables'] != 2 + 2 * receipt['trainable_tensors']):
            raise ValueError('Optimizer continuation proof missing')
    with np.load(run / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ('before', 'labels', 'ordinary', 'probe_before', 'unprotected'):
        if not np.array_equal(arrays[key], old[key]):
            raise ValueError('Original targets or labels changed')
    if (r['retention'] != retention(arrays['before'], arrays['after'], arrays['labels'], arrays['ordinary'])
            or arrays['probe_after'].shape != (22, 3) or not np.isfinite(arrays['probe_after']).all()
            or r['reviewed_background_correct'] != int((arrays['probe_after'].argmax(1) == 0).sum())
            or r['probe_identity'] != pr['probe_identity']
            or r['training_gate_pass'] != feasible(r, read_json(LOSS_PROTOCOL))):
        raise ValueError('Metrics differ from predictions')
    return run, p, r, arrays


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    trials = {}
    for arm in ('accepted', 'all_correct'):
        for seed in (47, 48):
            run, p, r, arrays = checked_run(arm, seed)
            trials[f'{arm}{seed}'] = dict(path=str(run), model_sha256=r['model_sha256'],
                predictions_sha256=r['predictions_sha256'], provenance_sha256=r['provenance_sha256'],
                parent_model_sha256=p['parent_model_sha256'], input_tensor_sha256=p['input_tensor_sha256'],
                retention=r['retention'], reviewed_background_correct=r['reviewed_background_correct'],
                reviewed_background_by_view={v: int((arrays['probe_after'][start:start + 11].argmax(1) == 0).sum())
                    for v, start in (('original', 0), ('blue_surface', 11))},
                training_gate_pass=r['training_gate_pass'])
    for seed in (47, 48):
        for key in ('parent_model_sha256', 'input_tensor_sha256'):
            if trials[f'accepted{seed}'][key] != trials[f'all_correct{seed}'][key]:
                raise ValueError('Ablation pair is not matched')
    eligible = all(trials[f'all_correct{s}']['training_gate_pass'] for s in (47, 48))
    result = dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), loss_protocol_sha256=sha256(LOSS_PROTOCOL), script_sha256=sha256(__file__),
        decision=('Both treatment seeds pass unchanged TRAIN feasibility; evaluate every paired model, not just the best seed.' if eligible else
                  'Do not export or evaluate development: both treatment seeds must pass unchanged TRAIN feasibility.'),
        scope='Matched TRAIN mask ablation. No held-out accuracy or live Pi evidence.',
        test_evaluated=False, development_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(dict(eligible_for_development=eligible, trials=trials))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
