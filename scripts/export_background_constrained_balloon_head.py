"""Export audited constrained heads with fixed FP32 runtime graph; no deployment."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.audit_background_constrained_balloon_head import OUTPUT as TRIAL, checked_fit
from scripts.fit_background_constrained_balloon_head import OUTPUT as FIT
from scripts.audit_blue_balloon_batch4 import checked_extra
from scripts.compare_current_blue_search import MODEL as PARENT, MODEL_SHA
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.grid_clutter_data import load_crop
from scripts.balloon_scene_constraints import scene_pixels


def eligible_names(trial):
    if (set(trial['results']) != {'control', 'admit'}
            or trial['results']['admit']['training_gate_pass'] is not True
            or trial['results']['admit'].get('all_background_correct') is not True
            or trial['eligible_for_full_scene_comparison'] is not True
            or any(trial[k] is not False for k in ('development_evaluated', 'test_evaluated',
                'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Require audited admission TRAIN gate')
    return ['control', 'admit']


def checked_trial():
    trial = read_json(TRIAL)
    names = eligible_names(trial)
    fit = checked_fit()
    if (trial['fit_sha256'] != sha256(FIT) or trial['results'] != fit['results']
            or trial['script_sha256'] != sha256('scripts/audit_background_constrained_balloon_head.py')
            or any(sha256(p) != h for p, h in trial['sources_sha256'].items())
            or sha256(PARENT) != MODEL_SHA):
        raise ValueError('Audited trial or parent changed')
    return trial, names


def export():
    trial, names = checked_trial()
    _, _, old, _, _, _ = checked_inputs()
    probes = np.stack([load_crop(r)[0] for r in checked_extra()])
    scene, _ = scene_pixels()
    x = np.concatenate([old, probes, scene])
    if x.shape != (2474, 64, 64, 3):
        raise ValueError('Parity inputs changed')
    parent = read_json(PARENT.with_suffix('.json'))
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
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            groups = [stored['after'], stored['probe_after'], stored['scene_after']]
        for pixels, expected in zip((old, probes, scene), groups, strict=True):
            if not np.allclose(model(pixels, training=False).numpy(), expected, atol=1e-5, rtol=0):
                raise ValueError('Checkpoint predictions changed')

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
        for pixels, expected in zip(x, np.concatenate(groups), strict=True):
            interpreter.set_tensor(inp['index'], pixels[None])
            interpreter.invoke()
            actual = interpreter.get_tensor(out['index'])[0]
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite export')
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            ap, ep = softmax(actual), softmax(expected)
            if (ap.argmax() != ep.argmax() or ((ap.argmax() > 0 and ap.max() >= .8)
                                               != (ep.argmax() > 0 and ep.max() >= .8))):
                raise ValueError('Export decision parity failed')
        if maximum > .002:
            raise ValueError('Export tensor parity failed')
        path.write_bytes(binary)
        contract = ('task', 'classes', 'size', 'layout', 'color', 'pixel_range', 'resize',
                    'normalization', 'output', 'threshold')
        meta = {k: parent[k] for k in contract}
        meta.update(sha256=sha256(path), bytes=len(binary), parameters=7763, tensor_dtype='float32',
            input_quantization=None, output_quantization=None, calibration_samples=0, calibration_split=None,
            integer_only_verified=False, synthetic_training=True,
            training_scope='Reviewed real TRAIN photos plus inherited synthetic color and reflection views',
            threshold_status='Fixed research0.8, not deployment calibration',
            source_student_sha256=row['model_sha256'], fitting_receipt_sha256=sha256(FIT),
            training_trial_sha256=sha256(TRIAL), export_script_sha256=sha256(__file__),
            parity_rows=len(x), max_keras_tflite_delta=maximum, decision_parity=True,
            experiment_role='matched_control' if name.startswith('control') else 'admission_candidate',
            deployment_approved=False, pi_zero_verified=False)
        write_json(path.with_suffix('.json'), meta)
        print(dict(name=name, bytes=len(binary), parity_rows=len(x), max_delta=maximum), flush=True)


if __name__ == '__main__':
    export()




