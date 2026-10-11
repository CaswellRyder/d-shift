"""Export paired models only after both admitted TRAIN gates pass."""
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from scripts.audit_wide_context import OUTPUT as TRIAL
from scripts.train_wide_context import inputs,PROTOCOL
from scripts.compare_current_blue_search import MODEL as PARENT,MODEL_SHA as PARENT_SHA


def export_path(name):
    if name not in ('control47','control48','admit47','admit48'):
        raise ValueError('Unknown export identity')
    return Path('runs/wide-context-exports-20261011')/name/'student.fp32.tflite'


def checked_trial():
    trial = read_json(TRIAL)
    if (trial['eligible_for_development'] is not True or trial['protocol_sha256']!=sha256(PROTOCOL)
            or trial['script_sha256']!=sha256('scripts/audit_wide_context.py')
            or sha256(PARENT)!=PARENT_SHA):
        raise ValueError('Require frozen paired TRAIN gate')
    for key,report in trial['trials'].items():
        run = Path(report['path'])
        prov = read_json(run/'provenance.json')
        if (any(sha256(p)!=h for p,h in prov['sources_sha256'].items())
                or (key.startswith('admit') and report['training_gate_pass'] is not True)):
            raise ValueError('Frozen training changed')
        for field,file in (('model_sha256','student.keras'),('predictions_sha256','predictions.npz'),('provenance_sha256','provenance.json')):
            if sha256(run/file)!=report[field]:
                raise ValueError('Checkpoint receipt changed')
    return trial


def export():
    trial = checked_trial()
    parent = read_json(PARENT.with_suffix('.json'))
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    for mode in ('control','admit'):
        _,_,_,tensors,_ = inputs('admit')
        x = np.concatenate(tensors)
        for seed in (47,48):
            report = trial['trials'][f'{mode}{seed}']
            run = Path(report['path'])
            path = export_path(f'{mode}{seed}')
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError(path)
            model = keras.models.load_model(run/'student.keras',compile=False)
            with np.load(run/'predictions.npz') as a:
                groups = [a[k+'after'] for k in ('','blue_','clutter_','real_','new_','runtime_')]
            expected = np.concatenate(groups)
            parameters = 8563 if mode=='admit' else 7763
            if model.count_params()!=parameters:
                raise ValueError('Architecture changed')
            # Reproduce the exact training-audit batch groups. Combining groups
            # can change floating-point reduction order despite identical weights.
            group_delta = 0.
            for tensor,reference in zip(tensors,groups,strict=True):
                actual = model(tensor,training=False).numpy()
                if not np.isfinite(actual).all():
                    raise ValueError('Nonfinite checkpoint output')
                group_delta = max(group_delta,float(np.max(np.abs(actual-reference))))
            if group_delta>1e-5:
                raise ValueError('Model does not reproduce original-group predictions')

            @tf.function(input_signature=[tf.TensorSpec([1,64,64,3],tf.float32)])
            def serve(x):
                return model(x,training=False)

            frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
            binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
            interpreter = tf.lite.Interpreter(model_content=binary,num_threads=1)
            interpreter.allocate_tensors()
            inp,out = interpreter.get_input_details()[0],interpreter.get_output_details()[0]
            delta = 0.
            for crop,reference in zip(x,expected,strict=True):
                interpreter.set_tensor(inp['index'],crop[None])
                interpreter.invoke()
                actual = interpreter.get_tensor(out['index'])[0]
                if not np.isfinite(actual).all():
                    raise ValueError('Nonfinite export')
                delta = max(delta,float(np.max(np.abs(actual-reference))))
            if delta>.002:
                raise ValueError('Export parity failed')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(binary)
            contract = ('task','classes','size','layout','color','pixel_range','resize','normalization','output','threshold')
            meta = {k:parent[k] for k in contract}
            meta.update(sha256=sha256(path),bytes=len(binary),parameters=parameters,tensor_dtype='float32',
                input_quantization=None,output_quantization=None,calibration_samples=0,calibration_split=None,
                integer_only_verified=False,synthetic_training=True,
                training_scope='Real TRAIN photographs plus inherited reviewed red-to-blue clutter augmentation',
                threshold_status='Fixed research0.8, not deployment calibration',source_student_sha256=report['model_sha256'],
                provenance_sha256=report['provenance_sha256'],training_trial_sha256=sha256(TRIAL),
                export_script_sha256=sha256(__file__),parity_rows=len(x),max_keras_tflite_delta=delta,
                max_checkpoint_original_group_delta=group_delta,
                deployment_approved=False,pi_zero_verified=False)
            write_json(path.with_suffix('.json'),meta)
            print(dict(mode=mode,seed=seed,bytes=len(binary),parity_rows=len(x),delta=delta),flush=True)


if __name__=='__main__':
    export()

