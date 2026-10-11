"""Verify initial probability equivalence, red invariance and builtin TFLite export."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_conditional_adapter import build
from scripts.train_blue_batch_trial import PROTOCOL, GROUPS, inputs

OUTPUT = Path('research/balloon-red-blue-20261008/blue-conditional-initialization-20261011.json')
RUN = Path('runs/blue-conditional-initialization-20261011')


def audit():
    if OUTPUT.exists() or RUN.exists():
        raise FileExistsError('Preserve prior initialization evidence')
    p = read_json(PROTOCOL)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    _, labels, _, tensors = inputs('admit')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    keras.utils.set_random_seed(47)
    parent = keras.models.load_model(p['parent'], compile=False)
    saved = [w.copy() for w in parent.get_weights()]
    model = build(parent)
    checks, references = [], []
    for name, x in zip(GROUPS, tensors, strict=True):
        a, b = softmax(parent(x, training=False).numpy()), softmax(model(x, training=False).numpy())
        delta = float(np.max(np.abs(a-b)))
        if (not np.isfinite(b).all() or delta > 1e-6
                or not np.array_equal(a.argmax(1), b.argmax(1))
                or not np.array_equal(a.max(1) >= .8, b.max(1) >= .8)):
            raise ValueError('Initial probabilities or decisions changed')
        checks.append(dict(group=name, rows=len(x), max_probability_delta=delta,
            tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        references.append(a)
    # Stress the final correction bias independently of any fitted labels.
    x, expected = np.concatenate(tensors), np.concatenate(references)
    layer = model.get_layer('blue_adapter_delta')
    kernel, bias = layer.get_weights()
    invariance = []
    for value in (-10., 10.):
        layer.set_weights([kernel, bias + value])
        actual = softmax(model(x, training=False).numpy())
        delta = float(np.max(np.abs(actual[:, 1] - expected[:, 1])))
        if delta > 1e-6 or not np.array_equal(actual[:, 1] >= .8, expected[:, 1] >= .8):
            raise ValueError('Red mass or acceptance changed')
        invariance.append(dict(correction=value, max_red_probability_delta=delta))
    layer.set_weights([kernel, bias])
    if any(not np.array_equal(a, b) for a, b in zip(saved, parent.get_weights(), strict=True)):
        raise ValueError('Parent weights modified')

    @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
    def serve(values):
        return model(values, training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    binary = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    export_delta = 0.
    for crop, reference in zip(x, expected, strict=True):
        interpreter.set_tensor(inp['index'], crop[None])
        interpreter.invoke()
        actual = softmax(interpreter.get_tensor(out['index'])[0])
        export_delta = max(export_delta, float(np.max(np.abs(actual-reference))))
        if (not np.isfinite(actual).all() or export_delta > .0001
                or actual.argmax() != reference.argmax()
                or (actual.max() >= .8) != (reference.max() >= .8)):
            raise ValueError('Export decisions or probabilities changed')
    RUN.mkdir()
    model.save(RUN / 'initial.keras')
    (RUN / 'initial.fp32.tflite').write_bytes(binary)
    impossible = int(((labels == 2) & (references[0][:, 1] > .2)).sum())
    write_json(OUTPUT, dict(parent=dict(path=p['parent'], sha256=p['parent_sha256']),
        checks=checks, red_invariance=invariance, total_probe_rows=len(x),
        parameters=model.count_params(), trainable_parameters=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
        blue_training_rows_with_parent_red_mass_above_point2=impossible,
        new_photo_probes_with_parent_red_mass_above_point2=int((references[-1][:, 1] > .2).sum()),
        max_export_probability_delta=export_delta, export_bytes=len(binary), builtin_only_conversion=True,
        initial_artifacts={str(p): sha256(p) for p in (RUN / 'initial.keras', RUN / 'initial.fp32.tflite')},
        sources_sha256={str(path): sha256(path) for path in (
            PROTOCOL, __file__, 'scripts/blue_conditional_adapter.py', 'scripts/blue_batch_trial_data.py',
            'scripts/blue_balloon_batch_data.py', 'scripts/train_blue_batch_trial.py')},
        limitations=['No learned correction or scene accuracy gain demonstrated.',
            'Red probability preservation does not guarantee unchanged downstream box selection.',
            'Frozen red mistakes cannot be corrected; blue probability is bounded by1-parent_red.',
            'Builtin conversion on desktop is not ARMv6 operator support or latency proof.'],
        next_action='Matched fixed TRAIN correction trials, then full-scene nonregression and separate ARMv6 execution checks.',
        trained=False, test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(rows=len(x), parameters=model.count_params(), bytes=len(binary), export_delta=export_delta,
        impossible_blue_train_rows=impossible, impossible_new_probes=int((references[-1][:, 1] > .2).sum())))


if __name__ == '__main__':
    audit()
