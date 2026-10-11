"""Export all paired final160 models only after both admitted TRAIN gates pass."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_blue_final_fit import checked_run, TRIAL, PROTOCOL

PARENT = Path('runs/balloon-mask-convergence-accepted-seed47-20261011/student.fp32.tflite')
PARENT_SHA = 'd31744368a38e9f27e2dc24285b121804b517c95ae38793fa37fc98f88522d79'


def checked_trial():
    trial = read_json(TRIAL)
    if (trial['eligible_for_development'] is not True
            or trial['protocol_sha256'] != sha256(PROTOCOL)
            or trial['script_sha256'] != sha256('scripts/balloon_blue_final_fit.py')
            or sha256(PARENT) != PARENT_SHA):
        raise ValueError('Require unchanged paired TRAIN gate and parent')
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, _, report, _, _ = checked_run(mode, seed)
            saved = trial['trials'][f'{mode}{seed}']
            if (saved['path'] != str(run)
                    or any(saved[k] != report[k] for k in ('model_sha256', 'predictions_sha256', 'provenance_sha256', 'training_gate_pass'))
                    or (mode == 'admit' and report['training_gate_pass'] is not True)):
                raise ValueError('Trial checkpoint changed')
    return trial


def export():
    checked_trial()
    parent = read_json(PARENT.with_suffix('.json'))
    if parent['sha256'] != PARENT_SHA or parent['threshold'] != .8:
        raise ValueError('Parent metadata changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, p, r, arrays, tensors = checked_run(mode, seed)
            path = run / 'student.fp32.tflite'
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError(path)
            model = keras.models.load_model(run / 'student.keras', compile=False)
            if model.count_params() != 7763:
                raise ValueError('Architecture changed')
            for tensor, key in zip(tensors, ('after', 'blue_after', 'probe_after'), strict=True):
                if not np.allclose(model(tensor, training=False).numpy(), arrays[key], rtol=0, atol=1e-5):
                    raise ValueError('Final checkpoint does not reproduce saved predictions')

            @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
            def serve(inputs):
                return model(inputs, training=False)

            frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
            binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
            interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
            interpreter.allocate_tensors()
            inp, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            x = np.concatenate(tensors)
            expected = np.concatenate([arrays[k] for k in ('after', 'blue_after', 'probe_after')])
            delta = 0.
            for crop, truth in zip(x, expected, strict=True):
                interpreter.set_tensor(inp['index'], crop[None])
                interpreter.invoke()
                actual = interpreter.get_tensor(output['index'])[0]
                if not np.isfinite(actual).all():
                    raise ValueError('Nonfinite export output')
                delta = max(delta, float(np.max(np.abs(actual - truth))))
            if delta > .002:
                raise ValueError('Export parity failed')
            path.write_bytes(binary)
            # Copy only the runtime contract, not inherited historical training claims.
            contract = ('task', 'classes', 'size', 'layout', 'color', 'pixel_range',
                        'resize', 'normalization', 'output', 'threshold')
            meta = {key: parent[key] for key in contract}
            meta.update(sha256=sha256(path), bytes=len(binary), tensor_dtype='float32',
                input_quantization=None, output_quantization=None,
                calibration_samples=0, calibration_split=None, integer_only_verified=False,
                synthetic_training=True, training_scope='Real photographs plus reviewed red-to-blue clutter augmentation; not synthetic-only',
                threshold_status='Fixed0.8 research operating point, not deployment calibration',
                source_checkpoint='student.keras', source_student_sha256=r['model_sha256'],
                provenance_sha256=r['provenance_sha256'], training=p, mode=mode, seed=seed,
                training_trial_sha256=sha256(TRIAL), export_script_sha256=sha256(__file__),
                max_keras_tflite_delta=delta, parity_rows=len(x), parameters=model.count_params(),
                deployment_approved=False, pi_zero_verified=False)
            write_json(path.with_suffix('.json'), meta)
            print(dict(mode=mode, seed=seed, bytes=len(binary), delta=delta, sha256=meta['sha256']))


if __name__ == '__main__':
    export()
