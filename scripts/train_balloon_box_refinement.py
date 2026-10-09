"""Fit a frozen-feature 516-parameter visible-balloon box head; research only."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import load_rgb, read_json, sha256, write_json
from dtr.tracking import iou
from scripts.train_balloon_quality import fit_ridge, fold_ids, training_rows


def check_config(config):
    expected = dict(ridge_penalty=10., holdout_fold=0, minimum_target_iou=.25,
        minimum_match_margin=.10, minimum_decoded_side=4., minimum_decoded_area_ratio=.25,
        maximum_decoded_area_ratio=4., minimum_decoded_overlap=.10)
    if (config.get('training_approved') is not True or config.get('deployment_approved') is not False
            or any(type(config.get(k)) not in (int, float) or config[k] != v for k, v in expected.items())):
        raise ValueError('Require fixed research-only box-refinement configuration')


def supervision(row, config):
    if row['kind'] != 'reviewed_balloon_overlap':
        return None, 'negative'
    ranked = sorted([(iou(row['box'], b), b) for b in row['target_boxes']], reverse=True)
    if not ranked or ranked[0][0] < config['minimum_target_iou']:
        return None, 'low_overlap'
    if len(ranked) > 1 and ranked[0][0]-ranked[1][0] < config['minimum_match_margin']:
        return None, 'ambiguous'
    target = ranked[0][1]
    a, b, c, d = row['crop_box']
    if not a <= target[0] < target[2] <= c or not b <= target[1] < target[3] <= d:
        return None, 'outside_crop'
    normalized = (np.asarray(target, np.float64)-[a, b, a, b])/[c-a, d-b, c-a, d-b]
    return dict(target_box=target, normalized_target=normalized.tolist()), 'admitted'


def decoded_box(values, proposal, config):
    """Bound correction to the visible crop; fail back to the original box."""
    values = np.asarray(values)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Invalid box-head output')
    values = np.clip(values, 0, 1)
    a, b, c, d = proposal['crop_box']
    box = (values*[c-a, d-b, c-a, d-b]+[a, b, a, b]).tolist()
    w, h = box[2]-box[0], box[3]-box[1]
    original = proposal['box']
    original_area = (original[2]-original[0])*(original[3]-original[1])
    if original_area <= 0:
        raise ValueError('Invalid original proposal area')
    if (min(w, h) < config['minimum_decoded_side']
            or not config['minimum_decoded_area_ratio'] <= w*h/original_area <= config['maximum_decoded_area_ratio']
            or iou(box, original) < config['minimum_decoded_overlap']):
        return list(original), False
    return box, True


def partition(rows, config):
    check_config(config)
    accepted, exclusions = [], Counter()
    for row in rows:
        target, reason = supervision(row, config)
        if target is None:
            exclusions[reason] += 1
        else:
            accepted.append(dict(row, **target))
    folds = fold_ids(accepted)
    train = [r for r, f in zip(accepted, folds) if f != config['holdout_fold']]
    hold = [r for r, f in zip(accepted, folds) if f == config['holdout_fold']]
    if len(train) < 2 or not hold:
        raise ValueError('Require training and held-out localization labels')
    for key in ('source_image', 'source_group', 'source_sha256'):
        if {r[key] for r in train} & {r[key] for r in hold}:
            raise ValueError('Cross-fold source leakage')
    return train, hold, dict(exclusions)


def box_metrics(rows, predictions, config):
    records = []
    for row, prediction in zip(rows, predictions):
        box, corrected = decoded_box(prediction, row, config)
        old, new = iou(row['box'], row['target_box']), iou(box, row['target_box'])
        records.append(dict(source_group=row['source_group'], proposal_box=row['box'], target_box=row['target_box'],
            predicted_box=box, correction_applied=corrected, original_iou=old, refined_iou=new))
    return dict(examples=records, count=len(records),
        mean_original_iou=float(np.mean([r['original_iou'] for r in records])),
        mean_refined_iou=float(np.mean([r['refined_iou'] for r in records])),
        original_localized=sum(r['original_iou'] >= .5 for r in records),
        refined_localized=sum(r['refined_iou'] >= .5 for r in records),
        improved=sum(r['refined_iou'] > r['original_iou'] for r in records),
        worsened=sum(r['refined_iou'] < r['original_iou'] for r in records),
        fallback=sum(not r['correction_applied'] for r in records))


def train(config_path, student_run, output):
    config, run, output = read_json(config_path), Path(student_run), Path(output)
    check_config(config)
    if output.exists():
        raise FileExistsError(output)
    rows, receipts = training_rows(read_json(config['quality_config']))
    fit_rows, hold_rows, excluded = partition(rows, config)
    meta = read_json(run/'student.fp32.json')
    checkpoint = run/'student.keras'
    if (meta.get('source_student_sha256') != sha256(checkpoint)
            or meta['sha256'] != sha256(run/'student.fp32.tflite')
            or meta['classes'] != ['background', 'red_balloon', 'blue_balloon']
            or meta['size'] != 64 or meta['task'] != 'balloon' or meta['synthetic_training']):
        raise ValueError('Frozen parent identity mismatch')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    pupil = keras.models.load_model(checkpoint, compile=False)
    pupil.trainable = False
    feature_model = keras.Model(pupil.input, pupil.get_layer('spatial_features').output)
    x = np.stack([load_rgb(r['absolute_path'], 64) for r in fit_rows])
    features = feature_model(x, training=False).numpy()
    y = np.asarray([r['normalized_target'] for r in fit_rows])
    regressors = [fit_ridge(features, y[:, k], config['ridge_penalty']) for k in range(4)]
    head = keras.layers.Dense(4, name='visible_box_xyxy')
    box = head(pupil.get_layer('spatial_features').output)
    model = keras.Model(pupil.input, keras.layers.Concatenate(name='class_logits_and_box')([pupil.output, box]))
    head.set_weights([np.stack([r[0] for r in regressors], axis=1), np.asarray([r[1] for r in regressors])])
    hx = np.stack([load_rgb(r['absolute_path'], 64) for r in hold_rows])
    all_x = np.concatenate([x, hx])
    expected = model(all_x, training=False).numpy()
    if not np.array_equal(expected[:, :3], pupil(all_x, training=False).numpy()):
        raise ValueError('Frozen class logits changed')
    output.mkdir(parents=True)
    model.save(output/'box.keras')

    @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
    def serve(inputs):
        return model(inputs, training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
    path = output/'box.fp32.tflite'
    path.write_bytes(binary)
    interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    results = []
    for crop in all_x:
        interpreter.set_tensor(inp['index'], crop[None])
        interpreter.invoke()
        results.append(interpreter.get_tensor(out['index'])[0])
    results = np.asarray(results)
    delta = float(np.abs(results-expected).max())
    if not np.isfinite(results).all() or delta > .002:
        raise ValueError('Box export parity failed')
    metadata = dict(kind='balloon_box_research', classes=meta['classes'], size=64, threshold=.8,
        output='first 3: class logits; final 4: crop-normalized visible-balloon xyxy', tensor_dtype='float32',
        sha256=sha256(path), bytes=len(binary), parameters=model.count_params(), added_parameters=head.count_params(),
        source_student_sha256=sha256(checkpoint), source_tflite_sha256=meta['sha256'],
        config=config, config_sha256=sha256(config_path), quality_config_sha256=sha256(config['quality_config']),
        source_reviews=receipts, script_sha256=sha256(__file__),
        quality_guard_sha256=sha256('scripts/train_balloon_quality.py'),
        training_samples=len(fit_rows), quality_label_holdout_samples=len(hold_rows),
        excluded=excluded, frozen_classifier=True, test_evaluated=False, synthetic_training=False,
        deployment_approved=False, pi_zero_verified=False)
    write_json(path.with_suffix('.json'), metadata)
    report = dict(training=box_metrics(fit_rows, results[:len(x), 3:], config),
        label_holdout=box_metrics(hold_rows, results[len(x):, 3:], config),
        holdout_caveat='Box labels excluded from head fit; original backbone saw related classification imagery',
        max_keras_tflite_delta=delta, test_evaluated=False, deployment_approved=False, pi_timing_measured=False)
    write_json(output/'report.json', report)
    print({k: {a:b for a,b in report[k].items() if a!='examples'} for k in ('training','label_holdout')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'student-run', 'output'):
        parser.add_argument(f'--{name}', required=True)
    args = parser.parse_args()
    train(args.config, args.student_run, args.output)
