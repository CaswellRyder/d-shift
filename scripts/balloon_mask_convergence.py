"""Fixed160 continuation and audit of both protection-mask arms, without data changes."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_all_correct import checked_run as parent_run
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.balloon_positive_protection import protected_loss, retention, feasible
from scripts.continue_balloon_protected_clutter import optimizer_evidence, LOSS_PROTOCOL
from scripts.grid_clutter_data import data, load_crop

PROTOCOL = Path('configs/balloon-mask-convergence-20261011.json')


def train(arm, seed):
    out = Path(f'runs/balloon-mask-convergence-{arm}-seed{seed}-20261011')
    protocol, settings = read_json(PROTOCOL), read_json(LOSS_PROTOCOL)
    if out.exists():
        raise FileExistsError(out)
    if arm not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Unknown arm or seed')
    parent, pp, pr, arrays = parent_run(arm, seed)
    rows, extra = data('admit')
    x = np.stack([load_crop(r)[0] for r in rows])
    if hashlib.sha256(x.tobytes()).hexdigest() != pp['input_tensor_sha256']:
        raise ValueError('Continuation inputs changed')
    labels, ordinary = arrays['labels'], arrays['ordinary']
    target = np.column_stack((np.eye(3)[labels], arrays['before'], ordinary)).astype(np.float32)
    probe = np.stack([load_crop(dict(r, view=v))[0] for v in ('original', 'blue_surface') for r in extra])
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    factory = all_correct_loss if arm == 'all_correct' else protected_loss
    loss = factory(settings['protection_weight'], settings['protection_probability_cap'], settings['parent_acceptance_threshold'])
    model = keras.models.load_model(parent / 'student.keras', custom_objects={'loss': loss})
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    restored = optimizer_evidence(model, 9984)
    if not np.allclose(model(x, training=False).numpy(), arrays['after'], rtol=0, atol=1e-5):
        raise ValueError('Checkpoint predictions differ')
    out.mkdir(parents=True)
    sources = (str(PROTOCOL), str(LOSS_PROTOCOL), 'scripts/audit_balloon_all_correct.py',
        'scripts/balloon_all_correct_protection.py', 'scripts/balloon_positive_protection.py',
        'scripts/continue_balloon_protected_clutter.py', 'scripts/grid_clutter_data.py', __file__)
    p = dict(arm=arm, seed=seed, parent=str(parent), parent_model_sha256=pr['model_sha256'],
        parent_provenance_sha256=pr['provenance_sha256'], parent_predictions_sha256=pr['predictions_sha256'],
        input_tensor_sha256=pp['input_tensor_sha256'], protocol_sha256=sha256(PROTOCOL),
        sources_sha256={p: sha256(p) for p in sources}, optimizer_restored=restored,
        parameters=7763, initial_epoch=104, final_epoch=160, original_parent_targets_preserved=True,
        dataset_shuffle_restarted=True, fixed_final_epoch=True, test_evaluated=False, deployment_approved=False)
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

        model.fit(dataset, initial_epoch=104, epochs=160, verbose=2,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        final_optimizer = optimizer_evidence(model, 15360)
        model.save(out / 'student.keras')
        after, probe_after = model(x, training=False).numpy(), model(probe, training=False).numpy()
        if not np.isfinite(after).all() or not np.isfinite(probe_after).all():
            raise ValueError('Nonfinite predictions')
        np.savez_compressed(out / 'predictions.npz', **dict(arrays, after=after, probe_after=probe_after))
        r = dict(retention=retention(arrays['before'], after, labels, ordinary),
            reviewed_background_correct=int((probe_after.argmax(1) == 0).sum()), reviewed_probe_rows=22,
            probe_identity=pr['probe_identity'], optimizer_final=final_optimizer,
            model_sha256=sha256(out / 'student.keras'), predictions_sha256=sha256(out / 'predictions.npz'),
            provenance_sha256=sha256(out / 'provenance.json'),
            scope='Fixed160 TRAIN continuation; not held-out accuracy', test_evaluated=False, deployment_approved=False)
        r['training_gate_pass'] = feasible(r, settings)
        if any(sha256(path) != h for path, h in p['sources_sha256'].items()):
            raise ValueError('Source changed during run')
        write_json(out / 'report.json', r)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, retention=r['retention'],
            background_correct=r['reviewed_background_correct'], training_gate_pass=r['training_gate_pass']))
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


def checked_run(arm, seed):
    parent, pp, pr, old = parent_run(arm, seed)
    run = Path(f'runs/balloon-mask-convergence-{arm}-seed{seed}-20261011')
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete' or p['arm'] != arm or p['seed'] != seed
            or p['parent'] != str(parent) or p['parent_model_sha256'] != pr['model_sha256']
            or p['parent_provenance_sha256'] != pr['provenance_sha256']
            or p['parent_predictions_sha256'] != pr['predictions_sha256']
            or p['input_tensor_sha256'] != pp['input_tensor_sha256'] or p['parameters'] != 7763
            or p['initial_epoch'] != 104 or p['final_epoch'] != 160 or p['protocol_sha256'] != sha256(PROTOCOL)
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())):
        raise ValueError('Continuation identities changed')
    for receipt, steps in ((p['optimizer_restored'], 9984), (r['optimizer_final'], 15360)):
        if (receipt['iterations'] != steps or receipt['finite_slots'] is not True or receipt['nonzero_slots'] is not True
                or not np.isclose(receipt['learning_rate'], .0001, rtol=0, atol=1e-10)
                or receipt['variables'] != 2 + 2 * receipt['trainable_tensors']):
            raise ValueError('Optimizer state proof missing')
    with np.load(run / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if any(not np.array_equal(arrays[k], old[k]) for k in ('before', 'labels', 'ordinary', 'probe_before', 'unprotected')):
        raise ValueError('Original targets changed')
    if (r['retention'] != retention(arrays['before'], arrays['after'], arrays['labels'], arrays['ordinary'])
            or arrays['probe_after'].shape != (22, 3) or not np.isfinite(arrays['probe_after']).all()
            or r['reviewed_background_correct'] != int((arrays['probe_after'].argmax(1) == 0).sum())
            or r['probe_identity'] != pr['probe_identity']
            or r['training_gate_pass'] != feasible(r, read_json(LOSS_PROTOCOL))):
        raise ValueError('Metrics do not reproduce')
    return run, p, r, arrays


def audit(output):
    if Path(output).exists():
        raise FileExistsError(output)
    trials = {}
    for arm in ('accepted', 'all_correct'):
        for seed in (47, 48):
            run, p, r, a = checked_run(arm, seed)
            trials[f'{arm}{seed}'] = dict(path=str(run), model_sha256=r['model_sha256'],
                predictions_sha256=r['predictions_sha256'], provenance_sha256=r['provenance_sha256'],
                retention=r['retention'], reviewed_background_correct=r['reviewed_background_correct'],
                training_gate_pass=r['training_gate_pass'],
                background_by_view={v: int((a['probe_after'][i:i + 11].argmax(1) == 0).sum()) for v, i in (('original', 0), ('blue_surface', 11))})
    eligible = all(trials[f'all_correct{s}']['training_gate_pass'] for s in (47, 48))
    write_json(output, dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        scope='Fixed160 paired TRAIN feasibility, not held-out or live Pi evidence',
        test_evaluated=False, development_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible, trials=trials))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('accepted', 'all_correct'))
    parser.add_argument('--seed', type=int, choices=(47, 48))
    parser.add_argument('--audit-output')
    args = parser.parse_args()
    if args.audit_output:
        audit(args.audit_output)
    elif args.arm and args.seed:
        train(args.arm, args.seed)
    else:
        parser.error('Specify arm/seed or audit-output')
