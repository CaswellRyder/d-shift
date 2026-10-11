"""Export all fixed160 paired checkpoints only after both treatments pass TRAIN."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_mask_convergence import checked_run
from scripts.grid_clutter_data import data, load_crop

TRIAL = Path('research/balloon-red-blue-20261008/mask-convergence-training-20261011.json')
PARENT = Path('runs/balloon-fragment-control-seed47-20261011')


def checked_trial():
    trial = read_json(TRIAL)
    if (trial['eligible_for_development'] is not True
            or trial['script_sha256'] != sha256('scripts/balloon_mask_convergence.py')
            or trial['protocol_sha256'] != sha256('configs/balloon-mask-convergence-20261011.json')):
        raise ValueError('Both treatment seeds must pass unchanged TRAIN gate')
    for arm in ('accepted', 'all_correct'):
        for seed in (47, 48):
            _, _, r, _ = checked_run(arm, seed)
            identity = trial['trials'][f'{arm}{seed}']
            if (any(r[k] != identity[k] for k in ('model_sha256', 'predictions_sha256', 'provenance_sha256', 'training_gate_pass'))
                    or (arm == 'all_correct' and r['training_gate_pass'] is not True)):
                raise ValueError('Trial checkpoint or gate changed')
    return trial


def export():
    checked_trial()
    rows, _ = data('admit')
    x = np.stack([load_crop(r)[0] for r in rows])
    parent = read_json(PARENT / 'color.fp32.json')
    if parent['sha256'] != sha256(PARENT / 'color.fp32.tflite'):
        raise ValueError('Parent metadata changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    for arm in ('accepted', 'all_correct'):
        for seed in (47, 48):
            run, p, r, arrays = checked_run(arm, seed)
            path = run / 'student.fp32.tflite'
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError(path)
            model = keras.models.load_model(run / 'student.keras', compile=False)
            expected = model(x, training=False).numpy()
            if model.count_params() != 7763 or not np.allclose(expected, arrays['after'], rtol=0, atol=1e-5):
                raise ValueError('Export checkpoint does not reproduce final predictions')

            @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
            def serve(inputs):
                return model(inputs, training=False)

            frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
            binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
            interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
            interpreter.allocate_tensors()
            inp, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            delta = 0.
            for crop, truth in zip(x, expected, strict=True):
                interpreter.set_tensor(inp['index'], crop[None])
                interpreter.invoke()
                actual = interpreter.get_tensor(output['index'])[0]
                if not np.isfinite(actual).all():
                    raise ValueError('Nonfinite export predictions')
                delta = max(delta, float(np.max(np.abs(actual - truth))))
            if delta > .002:
                raise ValueError('TFLite parity failed')
            path.write_bytes(binary)
            meta = dict(parent, sha256=sha256(path), bytes=len(binary), source_checkpoint='student.keras',
                source_student_sha256=r['model_sha256'], provenance_sha256=r['provenance_sha256'],
                training=p, training_trial_sha256=sha256(TRIAL), arm=arm, seed=seed,
                export_script_sha256=sha256(__file__), max_keras_tflite_delta=delta, parity_rows=len(x),
                deployment_approved=False, pi_zero_verified=False)
            for key in ('source_four_output_checkpoint_sha256', 'fragment_control_provenance_sha256',
                        'max_probe_keras_tflite_delta', 'probability_caveat'):
                meta.pop(key, None)
            write_json(path.with_suffix('.json'), meta)
            print(dict(arm=arm, seed=seed, bytes=len(binary), delta=delta, sha256=meta['sha256']))


if __name__ == '__main__':
    export()
