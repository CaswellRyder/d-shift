"""Export paired sparse localization adaptation after its TRAIN gate passes."""
import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_grid_blue_expansion import checked_run, OUTPUT as TRIAL
from scripts.train_grid_blue_expansion import PROTOCOL
from scripts.audit_balloon_dense_proposals import decode
from scripts.export_balloon_dense_proposals import proposal_parity


def checked_trial():
    trial = read_json(TRIAL)
    if (trial['eligible_for_development'] is not True
            or trial['protocol_sha256'] != sha256(PROTOCOL)
            or trial['script_sha256'] != sha256('scripts/audit_grid_blue_expansion.py')
            or any(sha256(p) != h for p, h in trial['helpers_sha256'].items())
            or not all(trial['trials'][f'admit{s}']['training_gate_pass'] for s in (47, 48))):
        raise ValueError('Require unchanged paired TRAIN gate')
    return trial


def export():
    trial = checked_trial()
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, _, probe, _, report, arrays = checked_run(mode, seed)
            if trial['trials'][f'{mode}{seed}']['report'] != report:
                raise ValueError('Audited checkpoint changed')
            path = run / 'proposal.fp32.tflite'
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError(path)
            model = keras.models.load_model(run / 'proposal.keras', compile=False)
            expected = model(probe['images'], training=False).numpy()
            if not np.allclose(expected, arrays['after'], rtol=0, atol=1e-5):
                raise ValueError('Checkpoint predictions changed')

            @tf.function(input_signature=[tf.TensorSpec([1, 120, 160, 3], tf.float32)])
            def serve(inputs):
                return model(inputs, training=False)

            frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
            binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
            runtime = tf.lite.Interpreter(model_content=binary, num_threads=1)
            runtime.allocate_tensors()
            inp, out = runtime.get_input_details()[0], runtime.get_output_details()[0]
            maximum = 0.
            for image, reference in zip(probe['images'], expected, strict=True):
                runtime.set_tensor(inp['index'], image[None])
                runtime.invoke()
                actual = runtime.get_tensor(out['index'])[0]
                if not np.isfinite(actual).all():
                    raise ValueError('Nonfinite export')
                maximum = max(maximum, float(np.max(np.abs(actual - reference))))
                proposal_parity(decode(actual), decode(reference))
            if maximum > .002:
                raise ValueError('Tensor parity failed')
            path.write_bytes(binary)
            metadata = dict(kind='balloon_dense_proposal_research', experiment='grid_blue_expansion', mode=mode, seed=seed,
                sha256=sha256(path), bytes=len(binary), parameters=model.count_params(),
                source_checkpoint_sha256=report['model_sha256'], provenance_sha256=report['provenance_sha256'],
                trial_sha256=sha256(TRIAL), input_shape=[1, 120, 160, 3], output_shape=[1, 15, 20, 6],
                tensor_dtype='float32', max_tensor_delta=maximum, decoded_parity_frames=156,
                preprocessing='320x240 RGB uint8 ->160x120 INTER_AREA float32 range0..255',
                proposal_threshold=.1, max_proposals=12, max_added_blue_proposals=2,
                decoder_sha256=sha256('scripts/audit_balloon_dense_proposals.py'), export_script_sha256=sha256(__file__),
                scope='Host tensor and decoded proposal parity. Not Pi compatibility or independent qualification.',
                test_evaluated=False, pi_tested=False, deployment_approved=False)
            write_json(path.with_suffix('.json'), metadata)
            print(dict(mode=mode, seed=seed, bytes=len(binary), delta=maximum, sha256=metadata['sha256']), flush=True)


if __name__ == '__main__':
    export()
