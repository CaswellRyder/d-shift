"""Export qualifying continuation arms with fixed-decoder parity on all TRAIN-source frames."""
import argparse

import numpy as np

from dtr.data import sha256, write_json, read_json
from scripts.audit_balloon_grid_convergence import checked_run
from scripts.audit_balloon_dense_proposals import decode as coarse_decode
from scripts.balloon_fine_grid import decode as fine_decode
from scripts.export_balloon_dense_proposals import proposal_parity


def export(arm, seed):
    trial_path = 'runs/balloon-grid-convergence-training-20261011.json'
    trial = read_json(trial_path)
    if arm not in trial['eligible_arms']:
        raise ValueError('Arm did not pass paired training gate')
    run, doc, x, _, _, _, report, _ = checked_run(arm, seed)
    expected_identity = trial['trials'][arm][str(seed)]
    if any(report[k] != expected_identity[k] for k in ('model_sha256', 'provenance_sha256', 'predictions_sha256')):
        raise ValueError('Training gate checkpoint changed')
    path = run / 'proposal.fp32.tflite'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model = keras.models.load_model(run / 'proposal.keras', compile=False)
    grid = (15, 20) if arm == 'coarse' else (30, 40)
    if model.count_params() != 2342 or model.output_shape != (None, *grid, 6):
        raise ValueError('Architecture changed')
    expected = model(x, training=False).numpy()
    if not np.allclose(expected, np.load(run / 'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0):
        raise ValueError('Checkpoint cannot reproduce saved predictions')

    @tf.function(input_signature=[tf.TensorSpec([1, 120, 160, 3], tf.float32)])
    def serve(inputs):
        return model(inputs, training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
    interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    decoder = coarse_decode if arm == 'coarse' else fine_decode
    maximum = 0.
    for image, reference in zip(x, expected, strict=True):
        interpreter.set_tensor(inp['index'], image[None])
        interpreter.invoke()
        actual = interpreter.get_tensor(out['index'])[0]
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite export')
        maximum = max(maximum, float(np.abs(actual - reference).max()))
        proposal_parity(decoder(actual), decoder(reference))
    if maximum > .002:
        raise ValueError('Tensor parity failed')
    path.write_bytes(binary)
    decoder_path = 'scripts/audit_balloon_dense_proposals.py' if arm == 'coarse' else 'scripts/balloon_fine_grid.py'
    metadata = dict(kind='balloon_grid_convergence_research', arm=arm, seed=seed,
        sha256=sha256(path), bytes=len(binary), parameters=model.count_params(),
        source_checkpoint_sha256=report['model_sha256'], provenance_sha256=report['provenance_sha256'],
        trial_sha256=sha256(trial_path), input_shape=[1, 120, 160, 3], output_shape=[1, *grid, 6],
        tensor_dtype='float32', max_tensor_delta=maximum, decoded_parity_frames=len(doc['frames']),
        preprocessing='320x240 RGB uint8 ->160x120 INTER_AREA float32 range0..255',
        proposal_threshold=.1, max_proposals=12, max_added_blue_proposals=2,
        decoder_path=decoder_path, decoder_sha256=sha256(decoder_path), export_script_sha256=sha256(__file__),
        scope='Host tensor and decoded proposal parity, not Pi compatibility or independent qualification',
        test_evaluated=False, pi_tested=False, deployment_approved=False)
    write_json(path.with_suffix('.json'), metadata)
    print(metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('coarse', 'context'), required=True)
    parser.add_argument('--seed', type=int, choices=(47, 48), required=True)
    args = parser.parse_args()
    export(args.arm, args.seed)
