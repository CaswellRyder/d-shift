"""Paired synthetic-data pilot. No deployment, reserved test, or Pi mutations.

Equal examples/optimizer steps: control repeats real crops in the extra slots;
intervention uses reviewed synthetic crops. No fake teacher logits for new data.
"""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, write_json, sha256, validate, load_rgb
from distill_pi_student import checked_cache, require_finite_metrics


def check_seed(path, classes):
    path = Path(path)
    doc = read_json(path)
    if doc.get('synthetic') is not True or doc.get('split') != 'train' or doc['classes'] != classes:
        raise ValueError('Require reviewed synthetic train-only matching taxonomy')
    if not 1 <= len(doc['samples']) <= 32:
        raise ValueError('Seed pilot requires 1..32 crops')
    for row in doc['samples']:
        crop = (path.parent/row['path']).resolve()
        if (row['split'] != 'train' or row.get('synthetic') is not True
                or not crop.is_relative_to(path.parent.resolve())
                or row['label'] not in classes or sha256(crop) != row['sha256']):
            raise ValueError('Invalid seed crop')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--seed-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    previous = read_json(args.base/'provenance.json')
    config = previous['config']
    manifest = Path(previous['manifest'])
    original = validate(manifest, config, splits=('train', 'val'))
    if original.get('synthetic') or original.get('training_approved') is not True:
        raise ValueError('Require approved real base data')
    seed = check_seed(args.seed_manifest, config['classes'])
    base_hash, manifest_hash = sha256(args.base/'student.keras'), sha256(manifest)
    if sha256(args.base/'student.float.tflite') != seed['model_sha256']:
        raise ValueError('Seed diagnostics did not use the base runtime model')
    rows = [r for r in original['samples'] if r['split'] == 'train']
    targets = checked_cache(args.base, manifest_hash, previous['teacher_sha256'], rows, config['classes'])
    k, size = len(config['classes']), config['student_size']
    targets = np.column_stack([targets, np.full(len(rows), config['alpha'], np.float32)])
    real = [(manifest.parent/r['path'], target) for r, target in zip(rows, targets)]
    extras = []
    for row in seed['samples']:
        target = np.zeros(2*k+1, np.float32)
        target[config['classes'].index(row['label'])] = 1
        target[-1] = 1  # Hard label only; teacher targets do not exist for new crops.
        extras.append((args.seed_manifest.parent/row['path'], target))
    extras *= 8
    rng = np.random.default_rng(config['seed'])
    control_extra = [real[i] for i in rng.integers(len(real), size=len(extras))]
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from dtr.data import batches
    from dtr.training import evaluate

    def dataset(samples):
        def generator():
            for path, target in samples:
                yield load_rgb(path, size), target
        ds = tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec((size, size, 3), tf.float32), tf.TensorSpec((2*k+1,), tf.float32)))
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(samples)))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return ds.shuffle(2048, seed=config['seed']).batch(config['batch_size']).with_options(options).prefetch(1)

    def loss(y, logits):
        hard = keras.losses.categorical_crossentropy(y[:, :k], logits, from_logits=True)
        temperature = config['temperature']
        old = y[:, k:2*k]/temperature
        soft = tf.reduce_sum(tf.nn.softmax(old)*(tf.nn.log_softmax(old)-tf.nn.log_softmax(
            logits/temperature)), axis=-1)*temperature*temperature
        return y[:, -1]*hard+(1-y[:, -1])*soft

    val = batches(manifest, config, 'val', size, validation_splits=('train', 'val'))
    sx = np.stack([load_rgb(args.seed_manifest.parent/r['path'], size) for r in seed['samples']])

    def fit_diagnostic(model):
        probabilities = tf.nn.softmax(model(sx, training=False)).numpy()
        return [dict(expected=r['label'], predicted=config['classes'][int(p.argmax())],
                     score=float(p.max())) for r, p in zip(seed['samples'], probabilities)]

    args.output.mkdir(parents=True)
    provenance = dict(base_model_sha256=base_hash, base_runtime_sha256=seed['model_sha256'],
                      manifest_sha256=manifest_hash, seed_manifest_sha256=sha256(args.seed_manifest),
                      script_sha256=sha256(__file__), epochs=2, learning_rate=1e-4,
                      extra_slots=len(extras), examples_per_epoch=len(real)+len(extras),
                      config=config, synthetic_source_scenes=2, same_optimizer_steps=True,
                      checkpoint_selection='fixed final epoch; no validation selection',
                      test_evaluated=False, deployment_approved=False)
    write_json(args.output/'provenance.json', provenance)
    write_json(args.output/'status.json', dict(state='running'))
    report = dict(scope='real crop validation; synthetic seed scores are training fit only', runs={})
    try:
        baseline = keras.models.load_model(args.base/'student.keras', compile=False)
        report['initial'] = dict(validation=evaluate(baseline, val, config['classes']),
                                 seed_fit=fit_diagnostic(baseline), parameters=baseline.count_params())
        del baseline
        for name, added in (('control', control_extra), ('synthetic', extras)):
            keras.backend.clear_session()
            keras.utils.set_random_seed(config['seed'])
            model = keras.models.load_model(args.base/'student.keras', compile=False)
            model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=loss)
            folder = args.output/name
            folder.mkdir()
            class Progress(keras.callbacks.Callback):
                def on_train_batch_end(self, batch, logs=None):
                    require_finite_metrics(logs)

                def on_epoch_end(self, epoch, logs=None):
                    write_json(args.output/'status.json', dict(state='running', run=name, epoch=epoch+1))
            model.fit(dataset(real+added), epochs=2, verbose=2,
                      callbacks=[Progress(), keras.callbacks.CSVLogger(str(folder/'epochs.csv'))])
            model.save(folder/'student.keras')
            report['runs'][name] = dict(validation=evaluate(model, val, config['classes']),
                                        seed_fit=fit_diagnostic(model), parameters=model.count_params(),
                                        model_sha256=sha256(folder/'student.keras'), synthetic_training=name=='synthetic',
                                        deployment_approved=False)
            write_json(args.output/'report.json', report)
        if sha256(args.base/'student.keras') != base_hash or sha256(manifest) != manifest_hash:
            raise ValueError('Base inputs changed')
        write_json(args.output/'status.json', dict(state='complete', deployment_approved=False))
        print({name: result['validation']['accuracy'] for name, result in report['runs'].items()})
    except BaseException as exc:
        write_json(args.output/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
