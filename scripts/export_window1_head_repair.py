"""Export independently audited frozen-feature heads; no runtime promotion."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.audit_window1_head_repair import OUTPUT as TRIAL, checked_fit
from scripts.fit_window1_head_repair import OUTPUT as FIT
from scripts.window1_priority_data import tensors
from scripts.compare_current_blue_search import MODEL as PARENT, MODEL_SHA


def checked_trial():
    trial, fit = read_json(TRIAL), checked_fit()
    if (trial['fit_sha256'] != sha256(FIT)
            or trial['script_sha256'] != sha256('scripts/audit_window1_head_repair.py')
            or any(sha256(p) != h for p, h in trial['helpers_sha256'].items())
            or set(trial['results']) != {'47', '48'}
            or trial['eligible_for_export_and_development'] is not True
            or not all(r['training_gate_pass'] for r in trial['results'].values())
            or sha256(PARENT) != MODEL_SHA
            or any(trial[k] is not False for k in ('development_evaluated', 'test_evaluated',
                                                  'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Require independently verified TRAIN repair')
    for arm, row in trial['results'].items():
        if row['model_sha256'] != fit['results'][arm]['model_sha256']:
            raise ValueError('Audited head changed')
    return trial, ['47', '48']


def export():
    trial, names = checked_trial()
    data, parent = tensors('priority'), read_json(PARENT.with_suffix('.json'))
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    for name in names:
        row = trial['results'][name]
        root = Path(row['run'])
        path = root/'student.fp32.tflite'
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
        model = keras.models.load_model(root/'student.keras', compile=False)
        if model.count_params() != 7763:
            raise ValueError('Architecture changed')
        with np.load(root/'constraints.npz', allow_pickle=False) as stored:
            base = stored['x']
            np.testing.assert_allclose(model(base, training=False).numpy(), stored['after'], atol=1e-5, rtol=0)
        x = np.concatenate([base, data['batch_pixels']])
        expected = model(x, training=False).numpy()

        @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
        def serve(image):
            return model(image, training=False)

        frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        binary = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        maximum = 0.
        for pixels, reference in zip(x, expected, strict=True):
            interpreter.set_tensor(inp['index'], pixels[None])
            interpreter.invoke()
            actual = interpreter.get_tensor(out['index'])[0]
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite exported prediction')
            maximum = max(maximum, float(np.max(np.abs(actual-reference))))
            ap, ep = softmax(actual), softmax(reference)
            if (ap.argmax() != ep.argmax() or ((ap.argmax() > 0 and ap.max() >= .8)
                                               != (ep.argmax() > 0 and ep.max() >= .8))):
                raise ValueError('Export decision parity failed')
        if maximum > .002:
            raise ValueError('Export tensor parity failed')
        path.write_bytes(binary)
        contract = ('task', 'classes', 'size', 'layout', 'color', 'pixel_range', 'resize',
                    'normalization', 'output', 'threshold')
        meta = {k:parent[k] for k in contract}
        meta.update(sha256=sha256(path), bytes=len(binary), parameters=7763, tensor_dtype='float32',
            input_quantization=None, output_quantization=None, calibration_samples=0, calibration_split=None,
            integer_only_verified=False, synthetic_training=True,
            training_scope='Reviewed real TRAIN photos with inherited synthetic color/reflection views; constrained head only',
            threshold_status='Fixed research0.8, not deployment calibration',
            source_student_sha256=row['model_sha256'], training_trial_sha256=sha256(TRIAL),
            fitting_receipt_sha256=sha256(FIT), export_script_sha256=sha256(__file__),
            parity_rows=len(x), max_keras_tflite_delta=maximum, decision_parity=True,
            experiment_role='repaired_priority_seed'+name,
            deployment_approved=False, pi_zero_verified=False)
        write_json(path.with_suffix('.json'), meta)
        print(dict(name=name, bytes=len(binary), parity_rows=len(x), max_delta=maximum), flush=True)
    checked_trial()


if __name__ == '__main__':
    export()

