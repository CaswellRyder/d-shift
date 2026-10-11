"""Export the color-only branch of a matched control; never a fragment model.

Dropping the fourth logit renormalizes probabilities, so new scene evaluation is
required. The training-only control never had a fourth-class positive label.
"""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from scripts.train_balloon_fragments import PARENT,CLASSES


def export(run):
    run=Path(run)
    provenance=read_json(run/'provenance.json')
    metadata=read_json(run/'fragment.fp32.json')
    if (provenance['mode']!='control' or provenance['classes']!=CLASSES
            or metadata['provenance_sha256']!=sha256(run/'provenance.json')
            or read_json(run/'status.json')['state']!='complete'):
        raise ValueError('Only completed color-label controls may drop the fragment output')
    path=run/'color.fp32.tflite'
    checkpoint=run/'color.keras'
    if path.exists() or checkpoint.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Color branch already exported')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    full=keras.models.load_model(run/'fragment.keras',compile=False)
    color=keras.Model(full.input,full.get_layer('classifier').output)
    if color.count_params()!=7763 or color.output_shape!=(None,3):
        raise ValueError('Standard color branch architecture changed')
    probe=np.random.default_rng(47).integers(0,256,(16,64,64,3)).astype(np.float32)
    expected=color(probe,training=False).numpy()
    if not np.array_equal(expected,full(probe,training=False).numpy()[:,:3]):
        raise ValueError('Color branch logits differ from control')
    color.save(checkpoint)

    @tf.function(input_signature=[tf.TensorSpec([1,64,64,3],tf.float32)])
    def serve(inputs):
        return color(inputs,training=False)

    frozen=convert_variables_to_constants_v2(serve.get_concrete_function())
    binary=tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
    interpreter=tf.lite.Interpreter(model_content=binary,num_threads=1)
    interpreter.allocate_tensors()
    inp,out=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
    delta=0.
    for x,y in zip(probe,expected):
        interpreter.set_tensor(inp['index'],x[None])
        interpreter.invoke()
        actual=interpreter.get_tensor(out['index'])[0]
        if not np.isfinite(actual).all():
            raise ValueError('Non-finite color export')
        delta=max(delta,float(np.max(np.abs(y-actual))))
    if delta>.002:
        raise ValueError('Color export parity failed')
    path.write_bytes(binary)
    parent=read_json(Path(PARENT)/'student.fp32.json')
    meta=dict(parent,sha256=sha256(path),bytes=len(binary),source_student_sha256=sha256(checkpoint),
        source_checkpoint='color.keras',source_four_output_checkpoint_sha256=sha256(run/'fragment.keras'),
        fragment_control_provenance_sha256=sha256(run/'provenance.json'),
        training=dict(provenance,manifest='data/balloon-red-blue-expansion2-admit-20261011/manifest.json',
            config=dict(parent['training']['config'],seed=provenance['seed']),
            classes_used_in_training=CLASSES,classes_exported=CLASSES[:3]),
        deployment_approved=False,pi_zero_verified=False,
        export_script_sha256=sha256(__file__),max_probe_keras_tflite_delta=delta,
        probability_caveat='Three-class normalization differs from four-output control; requires fresh full-scene evaluation')
    write_json(path.with_suffix('.json'),meta)
    print(dict(path=str(path),sha256=meta['sha256'],bytes=len(binary),parameters=color.count_params()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    export(p.parse_args().run)
