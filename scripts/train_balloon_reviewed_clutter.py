"""Matched training-only hard-clutter emphasis, same architecture and crop counts.

Start both arms at the same color-control checkpoint. Preserve its logits on
ordinary rows; reviewed negatives and matched background controls use hard labels.
No test images, checkpoint selection, camera access or deployment.
"""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import load_rgb, read_json, sha256, write_json
from scripts.train_balloon_fragments import checked_data, MANIFEST, QUEUE
from scripts.prepare_balloon_instance_review import checked_rows

PARENT = Path('runs/balloon-fragment-control-seed47-20261011')
REVIEW = 'configs/balloon-lab-clutter-review-20261011.json'
CLASSES = ['background', 'red_balloon', 'blue_balloon']


def data(mode):
    if mode not in ('control', 'admit'):
        raise ValueError('Require matched control or admit')
    ordinary, fragments = checked_data()
    negatives = checked_rows(MANIFEST, QUEUE, REVIEW)
    if len(negatives) != 9 or any(r['label'] != 'background' for r in negatives):
        raise ValueError('Require the nine explicitly reviewed negatives')
    known = {r['sha256']: r['label'] for r in ordinary}
    if any(r['sha256'] in known and known[r['sha256']] != 'background' for r in negatives):
        raise ValueError('Clutter conflicts with an existing positive label')
    rows = [dict(r, file=str(Path(MANIFEST).parent/r['path']), preserve=True) for r in ordinary]
    rows += [dict(r, label='blue_balloon', file=str(Path(QUEUE)/r['path']), preserve=True)
             for r in fragments]*4
    pools = [r for r in rows if r['label'] == 'background']
    rng = np.random.default_rng(47)
    additions = []
    for _ in range(4):
        for r in negatives:
            selected = (dict(r, file=str(Path(QUEUE)/r['path'])) if mode == 'admit'
                        else pools[int(rng.integers(len(pools)))])
            additions.append(dict(selected, preserve=False))
    return rows+additions, negatives


def loss_function():
    import tensorflow as tf

    def loss(target, logits):
        hard = tf.nn.softmax_cross_entropy_with_logits(labels=target[:, :3], logits=logits)
        teacher, pupil = target[:, 3:6]/4., logits/4.
        kl = tf.reduce_sum(tf.nn.softmax(teacher)*(tf.nn.log_softmax(teacher)-tf.nn.log_softmax(pupil)), axis=-1)*16.
        return hard+.5*target[:, 6]*kl

    return loss


def train(mode, seed, output, expansion=False, expansion4=False, hard_weight=1, epochs=8, openimages=False):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if seed not in (47, 48):
        raise ValueError('Require predeclared seed')
    if sum((expansion,expansion4,openimages)) > 1:
        raise ValueError('Choose one expansion experiment')
    if openimages:
        if hard_weight != 4 or epochs != 24:
            raise ValueError('Open Images requires the predeclared weight4, 24-epoch trial')
    elif hard_weight not in (1,16) or (hard_weight != 1 and not expansion4):
        raise ValueError('Hard weighting requires expansion4 and fixed weight 1 or 16')
    if not openimages and (epochs not in (8,80) or (epochs != 8 and not (expansion4 and hard_weight==16))):
        raise ValueError('Long convergence run requires expansion4, weight16 and fixed 80 epochs')
    if openimages:
        from scripts.openimages_balloon_training import data as expansion_data,QUEUE as queue,REVIEW as review
        rows, negatives = expansion_data(mode)
    elif expansion4:
        from scripts.balloon_expansion4 import data as expansion_data,QUEUE as queue,REVIEW as review
        rows, negatives = expansion_data(mode)
    elif expansion:
        from scripts.balloon_expansion3 import data as expansion_data,QUEUE as queue,REVIEW as review
        rows, negatives = expansion_data(mode)
    else:
        rows, negatives = data(mode)
        queue, review = QUEUE, REVIEW
    hard_rows = sum(not r['preserve'] for r in rows)
    parent = read_json(PARENT/'color.fp32.json')
    if (parent['sha256'] != sha256(PARENT/'color.fp32.tflite')
            or parent['source_student_sha256'] != sha256(PARENT/'color.keras')
            or parent['classes'] != CLASSES or parent['size'] != 64):
        raise ValueError('Parent identity changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(PARENT/'color.keras', compile=False)
    if model.count_params() != 7763 or model.output_shape != (None, 3):
        raise ValueError('Architecture changed')
    x = np.stack([load_rgb(r['file'], 64) for r in rows])
    labels = np.array([CLASSES.index(r['label']) for r in rows])
    old = model(x, training=False).numpy()
    targets = np.column_stack((np.eye(3)[labels], old, [r['preserve'] for r in rows])).astype(np.float32)
    out.mkdir(parents=True)
    provenance = dict(mode=mode, seed=seed, epochs=epochs, learning_rate=.0001, batch_size=16,
        manifest_sha256=sha256(MANIFEST), review_sha256=sha256(review),
        queue_sha256=sha256(Path(queue)/'review.json'), script_sha256=sha256(__file__),
        experiment='expansion4_hard' if expansion4 else ('expansion3' if expansion else 'reviewed_clutter'),
        data_helper_sha256=(sha256('scripts/balloon_expansion4.py') if expansion4 else
                           (sha256('scripts/balloon_expansion3.py') if expansion else None)),
        parent_model_sha256=parent['sha256'], parent_checkpoint_sha256=sha256(PARENT/'color.keras'),
        parent_metadata_sha256=sha256(PARENT/'color.fp32.json'),
        sample_identity=[{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve')} for r in rows],
        ordinary_rows=len(rows)-hard_rows, hard_label_rows=hard_rows,
        reviewed_unique_crops=len(negatives),
        reviewed_unique_negatives=sum(r['label']=='background' for r in negatives),
        hard_weight=hard_weight,
        fixed_final_epoch=True, classes=CLASSES, test_evaluated=False, deployment_approved=False)
    if epochs==80:
        provenance.update(convergence_protocol_sha256=sha256('configs/balloon-expansion4-convergence-20261011.json'),
                          trace_helper_sha256=sha256('scripts/balloon_convergence.py'))
    if openimages:
        provenance.update(experiment='openimages',
            data_helper_sha256=sha256('scripts/openimages_balloon_training.py'),
            trial_protocol_sha256=sha256('configs/balloon-openimages-trial-20261011.json'),
            license_verification_sha256=sha256('data/raw/openimages-balloon-train-20261011/license-verification.json'))
    write_json(out/'provenance.json', provenance)
    write_json(out/'status.json', dict(state='running'))
    try:
        tensors = (x, targets)
        if hard_weight != 1:
            weights = np.array([1. if r['preserve'] else float(hard_weight) for r in rows],np.float32)
            tensors = (x,targets,weights)
        ds = tf.data.Dataset.from_tensor_slices(tensors).shuffle(len(x), seed=seed).batch(16)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(.0001), loss=loss_function())
        trace = []
        if epochs==80:
            from scripts.balloon_convergence import snapshot
            probe = np.stack([load_rgb(Path(queue)/r['path'],64) for r in negatives])
            probe_labels = np.array([CLASSES.index(r['label']) for r in negatives])
            preserve = np.array([r['preserve'] for r in rows],dtype=bool)

        def record_probe(epoch):
            trace.append(snapshot(epoch, model(x,training=False).numpy(), labels, preserve,
                                  model(probe,training=False).numpy(), probe_labels))
            write_json(out/'training_trace.json',dict(hard_ids=[r['id'] for r in negatives],records=trace))

        class Progress(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                if epochs==80:
                    record_probe(0)

            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training metrics')
                write_json(out/'status.json', dict(state='running', epoch=epoch+1, metrics=logs))
                if epochs==80 and (epoch==0 or (epoch+1)%10==0):
                    record_probe(epoch+1)

        model.fit(ds, epochs=epochs, verbose=2, callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        model.save(out/'student.keras')
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
            delta = max(delta, float(np.max(np.abs(actual-expected))))
        if delta > .002 or sha256(MANIFEST) != provenance['manifest_sha256']:
            raise ValueError('Export parity or data identity changed')
        path = out/'student.fp32.tflite'
        path.write_bytes(binary)
        metadata = dict(parent, sha256=sha256(path), bytes=len(binary),
            source_student_sha256=sha256(out/'student.keras'), source_checkpoint='student.keras',
            training=provenance, provenance_sha256=sha256(out/'provenance.json'),
            deployment_approved=False, pi_zero_verified=False, max_keras_tflite_delta=delta)
        # Do not retain misleading provenance from the earlier branch export.
        for key in ('source_four_output_checkpoint_sha256', 'fragment_control_provenance_sha256',
                    'export_script_sha256', 'max_probe_keras_tflite_delta', 'probability_caveat'):
            metadata.pop(key, None)
        write_json(path.with_suffix('.json'), metadata)
        negative_logits = model(np.stack([load_rgb(Path(queue)/r['path'], 64) for r in negatives]), training=False).numpy()
        reviewed_labels = np.array([CLASSES.index(r['label']) for r in negatives])
        background = reviewed_labels==0
        report = dict(training_correct=int((final.argmax(1) == labels).sum()), training_rows=len(rows),
            ordinary_training_correct=int(((final.argmax(1)==labels)&np.array([r['preserve'] for r in rows])).sum()),
            ordinary_training_rows=sum(r['preserve'] for r in rows),
            reviewed_crop_correct=int((negative_logits.argmax(1)==reviewed_labels).sum()),
            reviewed_crop_count=len(negatives),
            reviewed_negative_background_correct=int((negative_logits.argmax(1)[background] == 0).sum()),
            reviewed_negative_count=int(background.sum()), max_keras_tflite_delta=delta,
            scope='Training fit only, not held-out validation', test_evaluated=False, deployment_approved=False)
        if epochs==80:
            report['training_trace_sha256'] = sha256(out/'training_trace.json')
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(state='complete'))
        print(report)
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('control', 'admit'), required=True)
    p.add_argument('--seed', type=int, choices=(47, 48), required=True)
    p.add_argument('--output', required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--expansion3',action='store_true',help='Use reviewed expansion3 views and matched class-count controls')
    group.add_argument('--expansion4',action='store_true',help='Use frozen-parent hard crops from reviewed expansion4 TRAIN views')
    group.add_argument('--openimages',action='store_true',help='Use reviewed licensed Open Images TRAIN crops')
    p.add_argument('--hard-weight',type=int,choices=(1,4,16),default=1,
                   help='Training-only learnability diagnostic; same weighting in matched controls')
    p.add_argument('--epochs',type=int,choices=(8,24,80),default=8)
    args = p.parse_args()
    train(args.mode, args.seed, args.output,args.expansion3,args.expansion4,args.hard_weight,args.epochs,args.openimages)
