"""Rebuild supervision, verify final checkpoints, export every fixed experiment arm."""
import csv
import hashlib

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_dense_proposals import decode
from scripts.blue_scene_localizer_data import PROTOCOL, data, dependencies, run_path
from scripts.export_balloon_dense_proposals import proposal_parity


def export(arm, seed):
    run = run_path(arm, seed)
    path = run/'proposal.fp32.tflite'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    protocol = read_json(PROTOCOL)
    x, y, folds, identities, groups = data(arm)
    p, r = read_json(run/'provenance.json'), read_json(run/'report.json')
    expected = dict(arm=arm, seed=seed, protocol_sha256=sha256(PROTOCOL), helpers_sha256=dependencies(),
        script_sha256=sha256('scripts/train_blue_scene_localizer.py'),
        parent_sha256=protocol['parent_sha256'], inputs_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        folds=folds.tolist(), additional_frames=identities, original_groups=groups,
        fit_frames=121, holdout_frames=16, optimizer_steps=3200, epochs=200,
        fresh_optimizer=True, fixed_final_checkpoint=True, parameters=2342,
        test_evaluated=False, deployment_approved=False, pi_tested=False)
    if (read_json(run/'status.json')['state'] != 'complete'
            or any(p.get(k) != v for k, v in expected.items())
            or r['provenance_sha256'] != sha256(run/'provenance.json')
            or r['model_sha256'] != sha256(run/'proposal.keras')
            or r['predictions_sha256'] != sha256(run/'predictions.npy')
            or p['targets_sha256'] != sha256(run/'targets.npy')
            or not np.array_equal(y, np.load(run/'targets.npy', allow_pickle=False))):
        raise ValueError('Run identity, schedule or reconstructed targets changed')
    with (run/'epochs.csv').open() as file:
        history = list(csv.DictReader(file))
    if ([int(row['epoch']) for row in history] != list(range(200))
            or not all(np.isfinite(float(row['loss'])) for row in history)):
        raise ValueError('Incomplete training history')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from scripts.balloon_dense_iou_loss import giou_loss
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model = keras.models.load_model(run/'proposal.keras', custom_objects={'giou_loss': giou_loss})
    if (model.count_params() != 2342 or model.output_shape != (None, 15, 20, 6)
            or int(model.optimizer.iterations) != 3200
            or not all(np.isfinite(v.numpy()).all() for v in model.optimizer.variables)):
        raise ValueError('Architecture or optimizer restoration failed')
    values = model(x, training=False).numpy()
    if not np.allclose(values, np.load(run/'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0):
        raise ValueError('Checkpoint predictions cannot be reproduced')
    for name, mask in (('fit_loss', folds != 0), ('holdout_loss', folds == 0)):
        value = float(giou_loss(tf.constant(y[mask]), tf.constant(values[mask])))
        if not np.isclose(value, r[name], atol=1e-5, rtol=0):
            raise ValueError('Loss report cannot be reproduced')

    @tf.function(input_signature=[tf.TensorSpec([1, 120, 160, 3], tf.float32)])
    def serve(inputs):
        return model(inputs, training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    binary = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    maximum = 0.
    for image, reference in zip(x, values, strict=True):
        interpreter.set_tensor(inp['index'], image[None])
        interpreter.invoke()
        actual = interpreter.get_tensor(out['index'])[0]
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite export')
        maximum = max(maximum, float(np.abs(actual-reference).max()))
        proposal_parity(decode(actual), decode(reference))
    if maximum > .002:
        raise ValueError('Export parity failed')
    path.write_bytes(binary)
    metadata = dict(kind='balloon_dense_proposal_research', experiment='blue_scene_localizer', arm=arm, seed=seed,
        sha256=sha256(path), bytes=len(binary), source_checkpoint_sha256=r['model_sha256'],
        provenance_sha256=r['provenance_sha256'], protocol_sha256=sha256(PROTOCOL), parameters=2342,
        input_shape=[1, 120, 160, 3], output_shape=[1, 15, 20, 6], tensor_dtype='float32',
        proposal_threshold=.1, max_proposals=12, max_added_blue_proposals=2,
        decoder_sha256=sha256('scripts/audit_balloon_dense_proposals.py'), export_script_sha256=sha256(__file__),
        preprocessing='320x240 RGB uint8 to160x120 INTER_AREA float32 range0..255',
        max_tensor_delta=maximum, decoded_parity_frames=len(x), test_evaluated=False,
        deployment_approved=False, pi_tested=False)
    write_json(path.with_suffix('.json'), metadata)
    parent = keras.models.load_model(protocol['parent'], compile=False)
    old = parent(x[-8:], training=False).numpy()
    diagnostics = dict(parent_negative_proposals=[len(decode(a)) for a in old],
        final_negative_proposals=[len(decode(a)) for a in values[-8:]],
        fit_loss=r['fit_loss'], holdout_loss=r['holdout_loss'])
    result = dict(metadata=metadata, diagnostics=diagnostics,
        epochs_sha256=sha256(run/'epochs.csv'), report_sha256=sha256(run/'report.json'),
        scope='Rebuilt TRAIN supervision and checkpoint/export parity; all arms evaluated without train-accuracy selection.')
    write_json(run/'audit.json', result)
    print(arm, seed, diagnostics, flush=True)
    return result


if __name__ == '__main__':
    from pathlib import Path
    output = Path('research/balloon-red-blue-20261008/blue-scene-localizer-training-20261011.json')
    if output.exists():
        raise FileExistsError(output)
    results = {f'{arm}{seed}': export(arm, seed) for arm in ('control', 'background') for seed in (47, 48)}
    write_json(output, dict(results=results, script_sha256=sha256(__file__), protocol_sha256=sha256(PROTOCOL),
        test_evaluated=False, deployment_approved=False, pi_tested=False))
