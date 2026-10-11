"""Fixed paired classifier trial on reviewed TRAIN clutter and synthetic blue surfaces."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.grid_clutter_data import data, load_crop, QUEUE, REVIEW
from scripts.train_balloon_fragments import MANIFEST
from scripts.train_balloon_reviewed_clutter import CLASSES, loss_function

PROTOCOL = Path('configs/balloon-grid-clutter-trial-20261011.json')
HELPER = 'scripts/grid_clutter_data.py'


def checked_protocol():
    protocol = read_json(PROTOCOL)
    preview = Path(protocol['preview'])
    parent = Path(protocol['parent'])
    if (sha256(HELPER) != protocol['data_helper_sha256']
            or sha256(REVIEW) != protocol['review_sha256']
            or sha256(parent / 'color.fp32.tflite') != protocol['parent_model_sha256']
            or sha256(preview / 'augmentation.json') != protocol['preview_receipt_sha256']
            or sha256(preview / 'crops-000.jpg') != protocol['preview_sheet_sha256']
            or protocol['deployment_approved'] is not False):
        raise ValueError('Predeclared trial or reviewed augmentation changed')
    augmentation = read_json(preview / 'augmentation.json')
    if (augmentation['script_sha256'] != sha256(HELPER)
            or augmentation['source_review_sha256'] != sha256(REVIEW)):
        raise ValueError('Synthetic preview does not match reviewed source')
    for sample in augmentation['samples']:
        if sha256(preview / sample['path']) != sample['sha256']:
            raise ValueError('Synthetic preview pixels changed')
    return protocol


def fit_counts(logits, labels, preserve):
    prediction = np.argmax(logits, axis=1)
    return {label: dict(correct=int(((prediction == labels) & preserve & (labels == i)).sum()),
                        total=int((preserve & (labels == i)).sum())) for i, label in enumerate(CLASSES)}


def train(mode, seed, output):
    protocol = checked_protocol()
    out, parent_path = Path(output), Path(protocol['parent'])
    if out.exists():
        raise FileExistsError(out)
    if mode not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Undeclared arm or seed')
    rows, extra = data(mode)
    preserve = np.array([r['preserve'] for r in rows], bool)
    if (int(preserve.sum()) != protocol['ordinary_rows']
            or int((~preserve).sum()) != protocol['hard_rows']):
        raise ValueError('Matched row counts changed')
    parent = read_json(parent_path / 'color.fp32.json')
    if (parent['sha256'] != protocol['parent_model_sha256']
            or parent['source_student_sha256'] != sha256(parent_path / 'color.keras')
            or parent['classes'] != CLASSES or parent['size'] != 64 or parent['threshold'] != .8):
        raise ValueError('Parent identity or operating point changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(parent_path / 'color.keras', compile=False)
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    loaded = [load_crop(r) for r in rows]
    x = np.stack([r[0] for r in loaded])
    fractions = [r[1] for r in loaded]
    labels = np.array([CLASSES.index(r['label']) for r in rows])
    old = model(x, training=False).numpy()
    targets = np.column_stack((np.eye(3)[labels], old, preserve)).astype(np.float32)
    probe_rows = [dict(r, view=view) for view in ('original', 'blue_surface') for r in extra]
    probe = np.stack([load_crop(r)[0] for r in probe_rows])
    before_probe = model(probe, training=False).numpy()
    out.mkdir(parents=True)
    sources = (str(PROTOCOL), HELPER, str(REVIEW), str(QUEUE / 'review.json'), str(MANIFEST),
               'scripts/train_balloon_reviewed_clutter.py', __file__)
    provenance = dict(mode=mode, seed=seed, protocol_sha256=sha256(PROTOCOL),
        sources_sha256={p: sha256(p) for p in sources}, parent_model_sha256=parent['sha256'],
        parent_checkpoint_sha256=sha256(parent_path / 'color.keras'),
        parent_metadata_sha256=sha256(parent_path / 'color.fp32.json'),
        sample_identity=[dict(source_image=r['source_image'], sha256=r['sha256'], label=r['label'],
            preserve=r['preserve'], view=r['view'], recolored_fraction=f) for r, f in zip(rows, fractions)],
        epochs=protocol['epochs'], learning_rate=protocol['learning_rate'], batch_size=protocol['batch_size'],
        ordinary_rows=int(preserve.sum()), hard_rows=int((~preserve).sum()),
        hard_weight=protocol['hard_weight'], final_epoch_only=True,
        test_evaluated=False, deployment_approved=False)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'status.json', dict(state='running'))
    try:
        weights = np.where(preserve, 1., protocol['hard_weight']).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((x, targets, weights))
        dataset = dataset.shuffle(len(x), seed=seed).batch(protocol['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(protocol['learning_rate']), loss=loss_function())

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training metrics')
                write_json(out / 'status.json', dict(state='running', epoch=epoch + 1, metrics=logs))

        model.fit(dataset, epochs=protocol['epochs'], verbose=2,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out / 'epochs.csv'))])
        model.save(out / 'student.keras')
        final = model(x, training=False).numpy()

        @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
        def serve(inputs):
            return model(inputs, training=False)

        frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
        binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
        interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
        interpreter.allocate_tensors()
        inp, result = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        delta = 0.
        for crop, expected in zip(x, final):
            interpreter.set_tensor(inp['index'], crop[None])
            interpreter.invoke()
            actual = interpreter.get_tensor(result['index'])[0]
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite export')
            delta = max(delta, float(np.max(np.abs(actual - expected))))
        if delta > .002 or any(sha256(p) != digest for p, digest in provenance['sources_sha256'].items()):
            raise ValueError('Export parity or source identity changed')
        path = out / 'student.fp32.tflite'
        path.write_bytes(binary)
        metadata = dict(parent, sha256=sha256(path), bytes=len(binary),
            source_student_sha256=sha256(out / 'student.keras'), source_checkpoint='student.keras',
            training=provenance, provenance_sha256=sha256(out / 'provenance.json'),
            deployment_approved=False, pi_zero_verified=False, max_keras_tflite_delta=delta)
        for key in ('source_four_output_checkpoint_sha256', 'fragment_control_provenance_sha256',
                    'export_script_sha256', 'max_probe_keras_tflite_delta', 'probability_caveat'):
            metadata.pop(key, None)
        write_json(path.with_suffix('.json'), metadata)
        after_probe = model(probe, training=False).numpy()
        report = dict(ordinary_before=fit_counts(old, labels, preserve),
            ordinary_after=fit_counts(final, labels, preserve),
            probe=[dict(id=r['id'], view=r['view'], label='background',
                before_logits=a.tolist(), after_logits=b.tolist(),
                before_prediction=CLASSES[int(a.argmax())], after_prediction=CLASSES[int(b.argmax())])
                for r, a, b in zip(probe_rows, before_probe, after_probe)],
            synthetic_rows=sum(r['view'] == 'blue_surface' for r in rows),
            mean_synthetic_changed_fraction=float(np.mean([f for r, f in zip(rows, fractions) if r['view'] == 'blue_surface'])),
            max_keras_tflite_delta=delta, parity_rows=len(x),
            scope='Training fit only, not held-out validation', test_evaluated=False, deployment_approved=False)
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete'))
        print(dict(mode=mode, seed=seed, ordinary=report['ordinary_after'],
                   probe_correct=sum(r['after_prediction'] == 'background' for r in report['probe']), delta=delta))
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('control', 'admit'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train(args.mode, args.seed, args.output)
