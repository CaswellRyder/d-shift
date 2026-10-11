"""Recompute the paired TRAIN feasibility gate from saved predictions and identities."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_positive_protection import retention, feasible
from scripts.grid_clutter_data import data
from scripts.train_balloon_protected_clutter import PROTOCOL
from scripts.train_balloon_reviewed_clutter import CLASSES


def checked_run(arm, seed):
    protocol = read_json(PROTOCOL)
    run = Path(f'runs/balloon-protected-clutter-{arm}-seed{seed}-20261011')
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete'
            or p['mode'] != arm or p['seed'] != seed or p['settings'] != protocol
            or p['protocol_sha256'] != sha256(PROTOCOL) or p['parameters'] != 7763
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())):
        raise ValueError('TRAIN run identity changed')
    rows, _ = data(arm)
    identity = [{k: row[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for row in rows]
    if p['sample_identity'] != identity:
        raise ValueError('TRAIN rows differ from audited inputs')
    with np.load(run / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    labels = np.array([CLASSES.index(row['label']) for row in rows])
    ordinary = np.array([row['preserve'] for row in rows], bool)
    if (not np.array_equal(arrays['labels'], labels) or not np.array_equal(arrays['ordinary'], ordinary)
            or arrays['probe_after'].shape != (22, 3) or not np.isfinite(arrays['probe_after']).all()
            or r['retention'] != retention(arrays['before'], arrays['after'], labels, ordinary)
            or r['unprotected_retention'] != retention(arrays['before'], arrays['unprotected'], labels, ordinary)
            or r['reviewed_background_correct'] != int((arrays['probe_after'].argmax(1) == 0).sum())
            or r['training_gate_pass'] != feasible(r, protocol)):
        raise ValueError('Metrics do not reproduce prediction arrays')
    return run, p, r, arrays


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    trials = {}
    for arm in ('control', 'admit'):
        for seed in (47, 48):
            run, p, r, _ = checked_run(arm, seed)
            trials[f'{arm}{seed}'] = dict(path=str(run),
                model_sha256=r['model_sha256'], predictions_sha256=r['predictions_sha256'],
                provenance_sha256=r['provenance_sha256'], input_tensor_sha256=p['input_tensor_sha256'],
                retention=r['retention'], unprotected_retention=r['unprotected_retention'],
                reviewed_background_correct=r['reviewed_background_correct'],
                training_gate_pass=r['training_gate_pass'])
    for arm in ('control', 'admit'):
        if trials[f'{arm}47']['input_tensor_sha256'] != trials[f'{arm}48']['input_tensor_sha256']:
            raise ValueError('Seeds trained on different tensors')
    passes = all(trials[f'admit{seed}']['training_gate_pass'] for seed in (47, 48))
    result = dict(trials=trials, eligible_for_development=passes,
        decision=('TRAIN feasibility passed; independent qualification still required.' if passes else
                  'Do not export or run development: positive protection reduces forgetting but does not yet learn sufficient reviewed clutter.'),
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        scope='TRAIN-only matched loss ablation. No held-out accuracy, Pi speed or readiness claim.',
        test_evaluated=False, development_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(dict(eligible_for_development=passes, trials=trials))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
