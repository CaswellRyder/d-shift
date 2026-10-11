"""Matched four-class research student: identical images, explicit fragment labels.

Control keeps the original blue color label for partial balloon crops; treatment
uses a separate fragment class. Both share the same architecture, initialization,
parent-logit preservation on ordinary training rows, seed and final epoch.
"""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import load_rgb,read_json,sha256,validate,write_json
from scripts.audit_balloon_search_stages import training_source
from scripts.prepare_balloon_instance_review import checked_rows

MANIFEST='data/balloon-red-blue-expansion2-admit-20261011/manifest.json'
QUEUE='data/balloon-lab-clutter-review-20261011'
REVIEW='configs/balloon-fragment-review-20261011.json'
PARENT='runs/balloon-red-blue-photometric-clutter4-admit-seed47-20261010'
CLASSES=['background','red_balloon','blue_balloon','balloon_fragment']


def fragment_anchor(row,frame):
    """Require reviewed blue parent and small crop coverage; no automatic labels."""
    x,y,z,w=row['crop_box']
    area=(z-x)*(w-y)
    matches=[]
    for target in frame['truth']:
        if target['label']!='blue_balloon':
            continue
        a,b,c,d=target['box']
        overlap=max(0,min(z,c)-max(x,a))*max(0,min(w,d)-max(y,b))
        if area>0 and overlap/area>=.85 and overlap/((c-a)*(d-b))<=.15:
            matches.append(target)
    if len(matches)!=1:
        raise ValueError('Fragment does not have one reviewed small-coverage blue parent')
    return matches[0]


def checked_data():
    config=read_json('configs/balloon-red-blue.json')
    base=validate(MANIFEST,config,splits=('train','val'))
    if base.get('synthetic',True) or base.get('training_approved') is not True:
        raise ValueError('Require approved real training data')
    rows=[r for r in base['samples'] if r['split']=='train']
    hold=[r for r in base['samples'] if r['split']!='train']
    for row in rows:
        training_source(row['source_image'])
        for key in ('source_image','source_sha256'):
            if row[key] in {r[key] for r in hold}:
                raise ValueError('Training source crosses reserved split')
    extra=checked_rows(MANIFEST,QUEUE,REVIEW,allow_fragments=True)
    queue,review=read_json(Path(QUEUE)/'review.json'),read_json(REVIEW)
    if review.get('control_color_label')!='blue_balloon' or len(extra)!=19:
        raise ValueError('Require fixed explicit fragment review')
    known={r['sha256'] for r in base['samples']}
    for row in extra:
        if row['label']!='balloon_fragment' or row['sha256'] in known:
            raise ValueError('Fragment label conflicts with original taxonomy')
        fragment_anchor(row,queue['frames'][row['frame_id']])
    return rows,extra


def expanded_student(parent):
    import keras
    kernel,bias=parent.get_layer('classifier').get_weights()
    if kernel.shape!=(128,3) or bias.shape!=(3,):
        raise ValueError('Require current three-class spatial head')
    head=keras.layers.Dense(1,name='fragment_classifier')
    fragment=head(parent.get_layer('spatial_features').output)
    model=keras.Model(parent.input,keras.layers.Concatenate(name='class_and_fragment_logits')([parent.output,fragment]))
    head.set_weights([kernel[:,2:3],np.array([bias[2]-6.],np.float32)])
    return model


def loss_function():
    import tensorflow as tf

    def loss(target,logits):
        hard=tf.nn.softmax_cross_entropy_with_logits(labels=target[:,:4],logits=logits)
        # Parent preservation only on ordinary rows, not on reviewed fragments.
        teacher=target[:,4:7]/4.
        pupil=logits[:,:3]/4.
        kl=tf.reduce_sum(tf.nn.softmax(teacher)*(tf.nn.log_softmax(teacher)-tf.nn.log_softmax(pupil)),axis=-1)*16.
        return hard+.5*target[:,7]*kl

    return loss


def train(mode,seed,output):
    if mode not in ('control','fragment') or seed not in (47,48):
        raise ValueError('Require fixed matched mode and seed')
    out=Path(output)
    if out.exists():
        raise FileExistsError(out)
    rows,extra=checked_data()
    parent_meta=read_json(Path(PARENT)/'student.fp32.json')
    parent_checkpoint=Path(PARENT)/'student.keras'
    if (parent_meta['source_student_sha256']!=sha256(parent_checkpoint)
            or parent_meta['sha256']!=sha256(Path(PARENT)/'student.fp32.tflite')
            or parent_meta['classes']!=CLASSES[:3] or parent_meta['size']!=64):
        raise ValueError('Parent identity changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    keras.utils.set_random_seed(seed)
    parent=keras.models.load_model(parent_checkpoint,compile=False)
    root=Path(MANIFEST).parent
    originals=np.stack([load_rgb(root/r['path'],64) for r in rows])
    fragments=np.stack([load_rgb(Path(QUEUE)/r['path'],64) for r in extra]*4)
    x=np.concatenate((originals,fragments))
    old_logits=parent(x,training=False).numpy()
    model=expanded_student(parent)
    ids=[CLASSES.index(r['label']) for r in rows]+[2 if mode=='control' else 3]*len(fragments)
    targets=np.column_stack((np.eye(4,dtype=np.float32)[ids],old_logits,
                             np.r_[np.ones(len(rows)),np.zeros(len(fragments))])).astype(np.float32)
    initial=model(x,training=False).numpy()
    if not np.array_equal(initial[:,:3],old_logits):
        raise ValueError('Initialization changed original logits')
    out.mkdir(parents=True)
    provenance=dict(mode=mode,seed=seed,epochs=8,batch_size=16,learning_rate=.0001,
        manifest_sha256=sha256(MANIFEST),queue_sha256=sha256(Path(QUEUE)/'review.json'),
        review_sha256=sha256(REVIEW),source_student_sha256=sha256(parent_checkpoint),
        source_tflite_sha256=parent_meta['sha256'],script_sha256=sha256(__file__),
        source_guard_sha256=sha256('scripts/prepare_balloon_instance_review.py'),
        ordinary_training_rows=len(rows),fragment_unique_crops=len(extra),fragment_repetitions=4,
        fragment_training_rows=len(fragments),classes=CLASSES,fragment_ids=[r['id'] for r in extra],
        sample_identity=[dict(source_image=r['source_image'],sha256=r['sha256']) for r in rows+extra],
        fixed_final_epoch=True,test_evaluated=False,deployment_approved=False)
    write_json(out/'provenance.json',provenance)
    write_json(out/'status.json',dict(state='running'))
    try:
        ds=tf.data.Dataset.from_tensor_slices((x,targets)).shuffle(len(x),seed=seed).batch(16)
        options=tf.data.Options()
        options.threading.private_threadpool_size=2
        ds=ds.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(.0001),loss=loss_function())

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Non-finite fragment training')
                write_json(out/'status.json',dict(state='running',epoch=epoch+1,metrics=logs))

        model.fit(ds,epochs=8,verbose=2,callbacks=[Progress(),keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        model.save(out/'fragment.keras')
        final=model(x,training=False).numpy()

        @tf.function(input_signature=[tf.TensorSpec([1,64,64,3],tf.float32)])
        def serve(inputs):
            return model(inputs,training=False)

        frozen=convert_variables_to_constants_v2(serve.get_concrete_function())
        binary=tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
        path=out/'fragment.fp32.tflite'
        path.write_bytes(binary)
        interpreter=tf.lite.Interpreter(model_content=binary,num_threads=1)
        interpreter.allocate_tensors()
        inp,res=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
        delta=0.
        for crop,expected in zip(x,final):
            interpreter.set_tensor(inp['index'],crop[None])
            interpreter.invoke()
            values=interpreter.get_tensor(res['index'])[0]
            if not np.isfinite(values).all():
                raise ValueError('Non-finite fragment export')
            delta=max(delta,float(np.max(np.abs(values-expected))))
        if delta>.002 or sha256(MANIFEST)!=provenance['manifest_sha256']:
            raise ValueError('Export parity or data identity changed')
        metadata=dict(kind='balloon_fragment_research',classes=CLASSES,size=64,threshold=.8,
            sha256=sha256(path),bytes=len(binary),parameters=model.count_params(),
            source_tflite_sha256=parent_meta['sha256'],provenance_sha256=sha256(out/'provenance.json'),
            output='Four class logits; fragment is never an accepted target',tensor_dtype='float32',
            test_evaluated=False,deployment_approved=False,pi_zero_verified=False)
        write_json(path.with_suffix('.json'),metadata)
        report=dict(max_keras_tflite_delta=delta,
            ordinary_training_correct=int(np.sum(final[:len(rows)].argmax(1)==np.array(ids[:len(rows)]))),
            ordinary_training_rows=len(rows),fragment_training_correct=int(np.sum(final[len(rows):].argmax(1)==ids[-1])),
            fragment_training_rows=len(fragments),test_evaluated=False,deployment_approved=False)
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(state='complete'))
        print(report)
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('control','fragment'),required=True)
    p.add_argument('--seed',type=int,choices=(47,48),required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    train(a.mode,a.seed,a.output)
