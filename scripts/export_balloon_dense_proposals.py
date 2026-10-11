"""Export sparse localizers; check tensor and decoded proposal parity on TRAIN data."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.train_balloon_dense_proposals import checked_data
from scripts.audit_balloon_dense_proposals import decode


def proposal_parity(actual,expected):
    if len(actual)!=len(expected):
        raise ValueError('Decoded proposal count changed')
    for a,b in zip(actual,expected,strict=True):
        if (a['color_group']!=b['color_group'] or a['crop_box']!=b['crop_box']
                or np.max(np.abs(np.asarray(a['box'])-b['box']))>.02
                or abs(a['proposal_score']-b['proposal_score'])>.001):
            raise ValueError('Decoded proposal geometry or ordering changed')


def export(root,run):
    run=Path(run)
    path=run/'proposal.fp32.tflite'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    doc,x,_,folds,_=checked_data(root)
    provenance,report=read_json(run/'provenance.json'),read_json(run/'report.json')
    if (read_json(run/'status.json')['state']!='complete'
            or provenance['manifest_sha256']!=sha256(Path(root)/'manifest.json')
            or provenance['folds']!=folds.tolist()
            or report['provenance_sha256']!=sha256(run/'provenance.json')
            or report['model_sha256']!=sha256(run/'proposal.keras')
            or report['predictions_sha256']!=sha256(run/'predictions.npy')):
        raise ValueError('Training source or checkpoint identity changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model=keras.models.load_model(run/'proposal.keras',compile=False)
    if model.count_params()!=2342 or model.output_shape!=(None,15,20,6):
        raise ValueError('Dense architecture changed')
    expected=model(x,training=False).numpy()
    if not np.allclose(expected,np.load(run/'predictions.npy',allow_pickle=False),atol=1e-5,rtol=0):
        raise ValueError('Checkpoint does not reproduce saved outputs')

    @tf.function(input_signature=[tf.TensorSpec([1,120,160,3],tf.float32)])
    def serve(inputs):
        return model(inputs,training=False)

    frozen=convert_variables_to_constants_v2(serve.get_concrete_function())
    binary=tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
    interpreter=tf.lite.Interpreter(model_content=binary,num_threads=1)
    interpreter.allocate_tensors()
    inp,out=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
    maximum=0.
    for image,reference in zip(x,expected,strict=True):
        interpreter.set_tensor(inp['index'],image[None])
        interpreter.invoke()
        actual=interpreter.get_tensor(out['index'])[0]
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite dense export')
        maximum=max(maximum,float(np.abs(actual-reference).max()))
        proposal_parity(decode(actual),decode(reference))
    if maximum>.002:
        raise ValueError('Dense tensor parity failed')
    path.write_bytes(binary)
    metadata=dict(kind='balloon_dense_proposal_research',sha256=sha256(path),bytes=len(binary),
        source_checkpoint_sha256=sha256(run/'proposal.keras'),provenance_sha256=sha256(run/'provenance.json'),
        manifest_sha256=sha256(Path(root)/'manifest.json'),parameters=model.count_params(),
        input_shape=[1,120,160,3],output_shape=[1,15,20,6],tensor_dtype='float32',
        channels=['red_logit','blue_logit','offset_x','offset_y','width','height'],
        preprocessing='OpenCV INTER_AREA from 320x240 RGB uint8 to 160x120 float32, range 0..255',
        proposal_threshold=.1,max_proposals=12,max_added_blue_proposals=2,
        decoder_sha256=sha256('scripts/audit_balloon_dense_proposals.py'),export_script_sha256=sha256(__file__),
        max_tensor_delta=maximum,decoded_parity_frames=len(doc['frames']),
        scope='TRAIN-frame export parity only; no Pi verification or independent qualification',
        test_evaluated=False,deployment_approved=False,pi_tested=False)
    write_json(path.with_suffix('.json'),metadata)
    print(dict(path=str(path),bytes=len(binary),max_tensor_delta=maximum,parity_frames=len(x)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--run',required=True)
    a=p.parse_args()
    export(a.data,a.run)
