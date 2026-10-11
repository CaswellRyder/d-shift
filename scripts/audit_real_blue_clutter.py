"""Recompute matched inputs, metrics and live checkpoint predictions for the trial."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.train_real_blue_clutter import PROTOCOL, run_path, inputs, calculate

TRIAL = Path('research/balloon-red-blue-20261008/real-blue-clutter-training-20261011.json')


def checked_run(mode, seed):
    run, protocol = run_path(mode, seed), read_json(PROTOCOL)
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete' or p['mode'] != mode or p['seed'] != seed
            or p['settings'] != protocol or p['parameters'] != 7763
            or sha256(protocol['parent']) != protocol['parent_sha256']
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())):
        raise ValueError('Trial identity changed')
    rows, labels, ordinary, tensors = inputs(mode)
    identity = [{k: row[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for row in rows]
    hashes = {k: hashlib.sha256(x.tobytes()).hexdigest() for k, x in zip(
        ('train', 'blue', 'clutter', 'real'), tensors, strict=True)}
    if p['sample_identity'] != identity or p['tensor_sha256'] != hashes:
        raise ValueError('Matched TRAIN pixels changed')
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    if (not np.array_equal(labels, arrays['labels']) or not np.array_equal(ordinary, arrays['ordinary'])
            or any(r[k] != value for k, value in calculate(arrays).items())):
        raise ValueError('Prediction metrics do not reproduce')
    return run, p, r, arrays, tensors


def audit():
    if TRIAL.exists():
        raise FileExistsError(TRIAL)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    protocol = read_json(PROTOCOL)
    parent = keras.models.load_model(protocol['parent'], compile=False)
    trials, first_rows, probe_hashes = {}, None, None
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, p, r, a, tensors = checked_run(mode, seed)
            common = p['sample_identity'][:1577]
            probes = {k: v for k, v in p['tensor_sha256'].items() if k != 'train'}
            if first_rows is not None and (first_rows != common or probe_hashes != probes):
                raise ValueError('Arms differ beyond new background rows')
            first_rows, probe_hashes = common, probes
            model = keras.models.load_model(run / 'student.keras', compile=False)
            if model.count_params() != 7763 or model.output_shape != (None, 3):
                raise ValueError('Architecture changed')
            delta = 0.
            for tensor, before, after in zip(tensors,
                    ('before', 'blue_before', 'clutter_before', 'real_before'),
                    ('after', 'blue_after', 'clutter_after', 'real_after'), strict=True):
                for current, expected in ((parent, a[before]), (model, a[after])):
                    actual = current(tensor, training=False).numpy()
                    if not np.isfinite(actual).all():
                        raise ValueError('Nonfinite checkpoint outputs')
                    delta = max(delta, float(np.max(np.abs(actual - expected))))
            if delta > 1e-5:
                raise ValueError('Checkpoint outputs do not reproduce saved evidence')
            trials[f'{mode}{seed}'] = dict(r, path=str(run), tensor_sha256=p['tensor_sha256'],
                verified_checkpoint_delta=delta, verified_checkpoint_rows=sum(len(t) for t in tensors))
        if trials[f'{mode}47']['tensor_sha256'] != trials[f'{mode}48']['tensor_sha256']:
            raise ValueError('Seeds trained on different tensors')
    eligible = all(trials[f'admit{s}']['training_gate_pass'] for s in (47, 48))
    write_json(TRIAL, dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        scope='TRAIN feasibility with fresh checkpoint readback; not development or Pi accuracy',
        test_evaluated=False, development_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible,
        trials={k: (v['new_blue_accepted'], v['reviewed_background_correct'], v['real_blue_background_correct']) for k, v in trials.items()}))


if __name__ == '__main__':
    audit()
