"""Fixed240 paired real-clutter continuation with immutable parent targets."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_real_blue_clutter import checked_run as parent_run
from scripts.train_real_blue_clutter import PROTOCOL as SETTINGS, calculate
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.continue_balloon_protected_clutter import optimizer_evidence

PROTOCOL = Path('configs/real-blue-clutter-convergence-20261011.json')
TRIAL = Path('research/balloon-red-blue-20261008/real-blue-clutter-convergence-20261011.json')


def run_path(mode, seed):
    protocol = read_json(PROTOCOL)
    if mode not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Undeclared mode or seed')
    return Path(f'runs/real-blue-clutter-convergence-{mode}-seed{seed}-20261011')


def train(mode, seed):
    out, settings = run_path(mode, seed), read_json(SETTINGS)
    if out.exists():
        raise FileExistsError(out)
    parent, pp, pr, arrays, tensors = parent_run(mode, seed)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    loss = all_correct_loss(settings['protection_weight'], settings['protection_probability_cap'], settings['parent_acceptance_threshold'])
    model = keras.models.load_model(parent / 'student.keras', custom_objects={'loss': loss})
    restored = optimizer_evidence(model, 8320)
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    names = ('after', 'blue_after', 'clutter_after', 'real_after')
    for tensor, key in zip(tensors, names, strict=True):
        if not np.allclose(model(tensor, training=False).numpy(), arrays[key], rtol=0, atol=1e-5):
            raise ValueError('Parent checkpoint does not reproduce')
    target = np.column_stack((np.eye(3)[arrays['labels']], arrays['before'], arrays['ordinary'])).astype(np.float32)
    sources = [str(PROTOCOL), str(SETTINGS), 'scripts/audit_real_blue_clutter.py',
        'scripts/train_real_blue_clutter.py', 'scripts/blue_false_alarm_data.py',
        'scripts/balloon_all_correct_protection.py', 'scripts/continue_balloon_protected_clutter.py', __file__]
    p = dict(mode=mode, seed=seed, parent=str(parent), parent_model_sha256=pr['model_sha256'],
        parent_provenance_sha256=pr['provenance_sha256'], parent_predictions_sha256=pr['predictions_sha256'],
        sources_sha256={path: sha256(path) for path in sources}, protocol_sha256=sha256(PROTOCOL),
        tensor_sha256=pp['tensor_sha256'], optimizer_restored=restored, parameters=7763,
        original_targets_preserved=True, dataset_shuffle_restarted=True, initial_epoch=80, final_epoch=240,
        test_evaluated=False, deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', p)
    write_json(out / 'status.json', dict(state='running'))
    try:
        weights = np.where(arrays['ordinary'], 1., settings['hard_weight']).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((tensors[0], target, weights)).shuffle(len(target), seed=seed).batch(16)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite metrics')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, initial_epoch=80, epochs=240, verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        final_optimizer = optimizer_evidence(model, 24960)
        result = dict(arrays)
        result.update({key: model(x, training=False).numpy() for key, x in zip(names, tensors, strict=True)})
        r = calculate(result)
        model.save(out / 'student.keras')
        np.savez_compressed(out / 'predictions.npz', **result)
        r.update(optimizer_final=final_optimizer, model_sha256=sha256(out / 'student.keras'),
            predictions_sha256=sha256(out / 'predictions.npz'), provenance_sha256=sha256(out / 'provenance.json'),
            scope='Fixed240 paired TRAIN convergence, not qualification', test_evaluated=False, deployment_approved=False)
        if any(sha256(path) != h for path, h in p['sources_sha256'].items()):
            raise ValueError('Source changed during training')
        write_json(out / 'report.json', r)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(mode=mode, seed=seed, report=r))
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


def checked_run(mode, seed):
    parent, pp, pr, old, tensors = parent_run(mode, seed)
    run = run_path(mode, seed)
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete' or p['mode'] != mode or p['seed'] != seed
            or p['parent'] != str(parent) or p['parent_model_sha256'] != pr['model_sha256']
            or p['parent_provenance_sha256'] != pr['provenance_sha256']
            or p['parent_predictions_sha256'] != pr['predictions_sha256']
            or p['tensor_sha256'] != pp['tensor_sha256'] or p['parameters'] != 7763
            or p['protocol_sha256'] != sha256(PROTOCOL) or p['initial_epoch'] != 80 or p['final_epoch'] != 240
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())):
        raise ValueError('Continuation identity changed')
    for receipt, steps in ((p['optimizer_restored'], 8320), (r['optimizer_final'], 24960)):
        if (receipt['iterations'] != steps or receipt['finite_slots'] is not True
                or receipt['nonzero_slots'] is not True
                or not np.isclose(receipt['learning_rate'], .0001, rtol=0, atol=1e-10)
                or receipt['variables'] != 2 + 2 * receipt['trainable_tensors']):
            raise ValueError('Optimizer state proof changed')
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    unchanged = set(old) - {'after', 'blue_after', 'clutter_after', 'real_after'}
    if any(not np.array_equal(arrays[k], old[k]) for k in unchanged) or any(r[k] != v for k, v in calculate(arrays).items()):
        raise ValueError('Original targets or recomputed metrics differ')
    return run, p, r, arrays, tensors


def audit():
    if TRIAL.exists():
        raise FileExistsError(TRIAL)
    trials = {}
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, p, r, _, _ = checked_run(mode, seed)
            trials[f'{mode}{seed}'] = dict(r, path=str(run), tensor_sha256=p['tensor_sha256'])
    eligible = all(trials[f'admit{s}']['training_gate_pass'] for s in (47, 48))
    write_json(TRIAL, dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        test_evaluated=False, development_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible,
        trials={k: (v['new_blue_accepted'], v['reviewed_background_correct'], v['real_blue_background_correct']) for k, v in trials.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('control', 'admit'))
    parser.add_argument('--seed', type=int, choices=(47, 48))
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.mode and args.seed:
        train(args.mode, args.seed)
    else:
        parser.error('Specify mode/seed or audit')
