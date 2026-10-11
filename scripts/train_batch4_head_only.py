"""Matched frozen-backbone classifier refinement on batch4; TRAIN-only."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_blue_balloon_batch4 import checked_extra, REVIEW
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.grid_clutter_data import load_crop
from scripts.train_batch3_retraining import LABELS, choose_controls, calculate

PROTOCOL = Path('configs/batch4-head-only-20261011.json')
QUEUE = Path('data/blue-balloon-batch4-review-20261011')
ROOT = Path('data/raw/openimages-blue-batch4-20261011')


def frozen_weights(model):
    return [v.copy() for layer in model.layers if layer.name != 'classifier' for v in layer.get_weights()]


def fold_head(model, head, frozen):
    if not all(np.array_equal(a, b) for a, b in zip(frozen, frozen_weights(model), strict=True)):
        raise ValueError('Frozen feature weights changed')
    model.get_layer('classifier').set_weights(head.get_layer('classifier').get_weights())


def train(arm, seed):
    p = read_json(PROTOCOL)
    if arm not in p['arms'] or seed not in p['seeds']:
        raise ValueError('Undeclared arm or seed')
    out = Path(f'runs/batch4-head-only-{arm}-seed{seed}-20261011')
    if out.exists():
        raise FileExistsError(out)
    rows, labels, old_pixels, _, _, previous = checked_inputs()
    extra = checked_extra()
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    if {r['source_image'] for r in extra} & {r['source_image'] for r in rows}:
        raise ValueError('New sources overlap existing curriculum')
    controls = choose_controls(rows, extra)
    # Both arms use the same class-block order, count, shuffle seed and weighting.
    selected = sorted(extra if arm == 'admit' else controls, key=lambda r: LABELS.index(r['label']))
    added = [dict(r, preserve=False, view='original') for _ in range(4) for r in selected]
    x = np.concatenate([old_pixels, np.stack([load_crop(r)[0] for r in added])])
    y = np.concatenate([labels, [LABELS.index(r['label']) for r in added]])
    ordinary = np.array([r['preserve'] for r in rows]+[False]*len(added), bool)
    probes = np.stack([load_crop(r)[0] for r in extra])
    probe_labels = np.array([LABELS.index(r['label']) for r in extra])
    steps = int(np.ceil(len(x)/p['batch_size']))*p['epochs']
    if len(x) != p['rows'] or steps != p['optimizer_steps'] or len(rows) != 2074:
        raise ValueError('Matched schedule or curriculum changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_all_correct_protection import all_correct_loss
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'], compile=False)
    if model.count_params() != p['parameters'] or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    teacher = model(x, training=False).numpy()
    probe_before = model(probes, training=False).numpy()
    feature_model = keras.Model(model.inputs, model.get_layer('spatial_features').output)
    features = feature_model(x, training=False).numpy()
    old_features = feature_model(old_pixels, training=False).numpy()
    probe_features = feature_model(probes, training=False).numpy()
    if features.shape != (len(x), 128) or not np.isfinite(features).all():
        raise ValueError('Unexpected frozen feature tensor')
    frozen = frozen_weights(model)
    feature_input = keras.Input(shape=(128,))
    head = keras.Model(feature_input, keras.layers.Dense(3, name='classifier')(feature_input))
    head.get_layer('classifier').set_weights(model.get_layer('classifier').get_weights())
    if head.count_params() != p['trainable_parameters']:
        raise ValueError('Trainable head changed')
    target = np.column_stack([np.eye(3)[y], teacher, ordinary]).astype(np.float32)
    paths = [PROTOCOL, REVIEW, QUEUE/'review.json', ROOT/'acquisition.json', ROOT/'license-verification.json',
        previous, __file__, 'scripts/audit_blue_balloon_batch4.py', 'scripts/reviewed_balloon_batch.py', 'scripts/train_batch3_retraining.py', 'scripts/fit_blue_prototype_rejection.py', 'scripts/train_batch4_retraining.py',
        'scripts/balloon_all_correct_protection.py', 'scripts/balloon_positive_protection.py',
        'research/balloon-red-blue-20261008/blue-balloon-batch4-admission-20261011.json']
    sources = {str(path): sha256(path) for path in paths}
    provenance = dict(arm=arm, seed=seed, settings=p, sources_sha256=sources,
        tensor_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in (x, y, ordinary, probes, probe_labels)],
        sample_identity=[{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows+added],
        parameters=model.count_params(), trainable_parameters=head.count_params(),
        feature_sha256=hashlib.sha256(features.tobytes()).hexdigest(), fixed_final_epoch=True, fresh_optimizer=True,
        development_evaluated=False, test_evaluated=False, deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out/'provenance.json', provenance)
    write_json(out/'status.json', dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((features, target, np.where(ordinary, 1., p['hard_weight']).astype(np.float32)))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.shuffle(len(x), seed=seed).batch(p['batch_size']).with_options(options).prefetch(1)
        schedule = keras.optimizers.schedules.CosineDecay(p['learning_rate'], steps, alpha=p['cosine_alpha'])
        head.compile(optimizer=keras.optimizers.Adam(schedule), loss=all_correct_loss(
            p['protection_weight'], p['protection_probability_cap'], p['parent_acceptance_threshold']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if (epoch+1) % 20 == 0:
                    print(dict(arm=arm, seed=seed, epoch=epoch+1, metrics=logs), flush=True)

        head.fit(ds, epochs=p['epochs'], verbose=0,
            callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        fold_head(model, head, frozen)
        after = model(old_pixels, training=False).numpy()
        probe_after = model(probes, training=False).numpy()
        if int(head.optimizer.iterations) != steps or not np.isfinite(after).all() or not np.isfinite(probe_after).all():
            raise ValueError('Incomplete or nonfinite training')
        fold_delta = max(float(np.max(np.abs(after-head(old_features, training=False).numpy()))),
                         float(np.max(np.abs(probe_after-head(probe_features, training=False).numpy()))))
        if fold_delta > 1e-5:
            raise ValueError('Folded head differs from feature-only training')
        report = calculate(teacher[:2074], after, labels, probe_after, probe_labels)
        model.save(out/'student.keras')
        np.savez_compressed(out/'predictions.npz', before=teacher[:2074], after=after, labels=labels,
                            probe_before=probe_before, probe_after=probe_after, probe_labels=probe_labels)
        if any(sha256(path) != digest for path, digest in sources.items()):
            raise ValueError('Inputs changed during training')
        report.update(frozen_weights_unchanged=True, head_fold_max_delta=fold_delta, trainable_parameters=head.count_params(), model_sha256=sha256(out/'student.keras'), predictions_sha256=sha256(out/'predictions.npz'),
            provenance_sha256=sha256(out/'provenance.json'), optimizer_steps=steps,
            scope='TRAIN feasibility only, no independent accuracy, scene recall or Pi readiness claim.')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(dict(arm=arm, seed=seed, report=report), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'admit'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    a = parser.parse_args()
    train(a.arm, a.seed)
