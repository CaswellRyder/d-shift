"""Fixed80 continuation; keep original targets, optimizer and matched TRAIN rows."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_blue_expansion import checked_run as parent_run, counts
from scripts.train_balloon_blue_expansion import PROTOCOL as SETTINGS, training_gate
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.balloon_positive_protection import retention
from scripts.continue_balloon_protected_clutter import optimizer_evidence

PROTOCOL = Path('configs/balloon-blue-expansion2-convergence-20261011.json')


def train(mode, seed):
    protocol, settings = read_json(PROTOCOL), read_json(SETTINGS)
    out = Path(f'runs/balloon-blue-expansion2-convergence-{mode}-seed{seed}-20261011')
    if out.exists():
        raise FileExistsError(out)
    if mode not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Unknown arm or seed')
    parent, pp, pr, arrays, (x, blue, probes) = parent_run(mode, seed)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    loss = all_correct_loss(settings['protection_weight'], settings['protection_probability_cap'], settings['parent_acceptance_threshold'])
    model = keras.models.load_model(parent / 'student.keras', custom_objects={'loss': loss})
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    restored = optimizer_evidence(model, 2376)
    for tensor, expected in ((x, arrays['after']), (blue, arrays['blue_after']), (probes, arrays['probe_after'])):
        if not np.allclose(model(tensor, training=False).numpy(), expected, rtol=0, atol=1e-5):
            raise ValueError('Parent checkpoint does not reproduce')
    labels, ordinary = arrays['labels'], arrays['ordinary']
    target = np.column_stack((np.eye(3)[labels], arrays['before'], ordinary)).astype(np.float32)
    sources = [str(PROTOCOL), str(SETTINGS), 'scripts/audit_balloon_blue_expansion.py',
        'scripts/train_balloon_blue_expansion.py', 'scripts/openimages_blue_expansion.py',
        'scripts/balloon_all_correct_protection.py', 'scripts/balloon_positive_protection.py',
        'scripts/continue_balloon_protected_clutter.py', __file__]
    p = dict(mode=mode, seed=seed, parent=str(parent), parent_model_sha256=pr['model_sha256'],
        parent_provenance_sha256=pr['provenance_sha256'], parent_predictions_sha256=pr['predictions_sha256'],
        sources_sha256={path: sha256(path) for path in sources}, protocol_sha256=sha256(PROTOCOL),
        input_tensor_sha256=pp['input_tensor_sha256'], optimizer_restored=restored,
        original_parent_targets_preserved=True, dataset_shuffle_restarted=True,
        initial_epoch=24, final_epoch=80, parameters=7763, test_evaluated=False, deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', p)
    write_json(out / 'status.json', dict(state='running'))
    try:
        weights = np.where(ordinary, 1., settings['hard_weight']).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((x, target, weights)).shuffle(len(x), seed=seed).batch(16)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite metrics')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, initial_epoch=24, epochs=80, verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        final_optimizer = optimizer_evidence(model, 7920)
        after = model(x, training=False).numpy()
        blue_after, probe_after = model(blue, training=False).numpy(), model(probes, training=False).numpy()
        if not all(np.isfinite(a).all() for a in (after, blue_after, probe_after)):
            raise ValueError('Nonfinite outputs')
        model.save(out / 'student.keras')
        np.savez_compressed(out / 'predictions.npz', **dict(arrays, after=after, blue_after=blue_after, probe_after=probe_after))
        correct, accepted = counts(blue_after)
        r = dict(retention=retention(arrays['before'], after, labels, ordinary),
            warm_start_retention=retention(arrays['start'], after, labels, ordinary),
            reviewed_background_correct=int((probe_after.argmax(1) == 0).sum()),
            new_blue_correct=correct, new_blue_accepted=accepted, optimizer_final=final_optimizer,
            model_sha256=sha256(out / 'student.keras'), predictions_sha256=sha256(out / 'predictions.npz'),
            provenance_sha256=sha256(out / 'provenance.json'), scope='Fixed80 TRAIN convergence, not held-out or Pi qualification',
            test_evaluated=False, deployment_approved=False)
        r['training_gate_pass'] = training_gate(r, settings)
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
    run = Path(f'runs/balloon-blue-expansion2-convergence-{mode}-seed{seed}-20261011')
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete' or p['mode'] != mode or p['seed'] != seed
            or p['parent'] != str(parent) or p['parent_model_sha256'] != pr['model_sha256']
            or p['parent_provenance_sha256'] != pr['provenance_sha256']
            or p['parent_predictions_sha256'] != pr['predictions_sha256']
            or p['input_tensor_sha256'] != pp['input_tensor_sha256']
            or p['protocol_sha256'] != sha256(PROTOCOL) or p['initial_epoch'] != 24 or p['final_epoch'] != 80
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())):
        raise ValueError('Continuation identity changed')
    for receipt, steps in ((p['optimizer_restored'], 2376), (r['optimizer_final'], 7920)):
        if (receipt['iterations'] != steps or receipt['finite_slots'] is not True
                or receipt['nonzero_slots'] is not True
                or not np.isclose(receipt['learning_rate'], .0001, rtol=0, atol=1e-10)
                or receipt['variables'] != 2 + 2 * receipt['trainable_tensors']):
            raise ValueError('Optimizer evidence changed')
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        a = {k: saved[k] for k in saved.files}
    unchanged = set(old) - {'after', 'blue_after', 'probe_after'}
    correct, accepted = counts(a['blue_after'])
    if (any(not np.array_equal(a[k], old[k]) for k in unchanged)
            or r['retention'] != retention(a['before'], a['after'], a['labels'], a['ordinary'])
            or r['warm_start_retention'] != retention(a['start'], a['after'], a['labels'], a['ordinary'])
            or a['probe_after'].shape != (22, 3) or not np.isfinite(a['probe_after']).all()
            or r['reviewed_background_correct'] != int((a['probe_after'].argmax(1) == 0).sum())
            or r['new_blue_correct'] != correct or r['new_blue_accepted'] != accepted
            or r['training_gate_pass'] != training_gate(r, read_json(SETTINGS))):
        raise ValueError('Continuation metrics do not reproduce')
    return run, p, r, a, tensors


def audit(output):
    if Path(output).exists():
        raise FileExistsError(output)
    trials = {}
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, p, r, _, _ = checked_run(mode, seed)
            trials[f'{mode}{seed}'] = dict(r, path=str(run), input_tensor_sha256=p['input_tensor_sha256'])
    eligible = all(trials[f'admit{s}']['training_gate_pass'] for s in (47, 48))
    write_json(output, dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        test_evaluated=False, development_evaluated=False, deployment_approved=False,
        pi_tested=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible,
        trials={k: (v['new_blue_accepted'], v['reviewed_background_correct']) for k, v in trials.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('control', 'admit'))
    parser.add_argument('--seed', type=int, choices=(47, 48))
    parser.add_argument('--audit-output')
    args = parser.parse_args()
    if args.audit_output:
        audit(args.audit_output)
    elif args.mode and args.seed:
        train(args.mode, args.seed)
    else:
        parser.error('Specify mode/seed or audit-output')
