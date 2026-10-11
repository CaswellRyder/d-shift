"""Train a small separate proposal network with sparse, reviewed TRAIN labels.

The existing classifier is untouched. Grouped holdout is within known training
environments, not independent competition qualification. No development tuning.
"""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.audit_public_balloon_export import source_group
from scripts.prepare_balloon_dense_proposals import encode
from scripts.audit_balloon_search_stages import training_source


def network():
    import keras
    layers = keras.layers
    inputs = layers.Input((120,160,3))
    x = layers.Rescaling(1/255.)(inputs)
    x = layers.Conv2D(8,3,strides=2,padding='same',activation='relu')(x)
    x = layers.SeparableConv2D(16,3,strides=2,padding='same',activation='relu')(x)
    x = layers.SeparableConv2D(24,3,strides=2,padding='same',activation='relu')(x)
    x = layers.SeparableConv2D(24,5,padding='same',activation='relu')(x)
    heat = layers.Conv2D(2,1,bias_initializer=keras.initializers.Constant(-4.),name='center_logits')(x)
    box = layers.Conv2D(4,1,activation='sigmoid',name='center_geometry')(x)
    return keras.Model(inputs,layers.Concatenate(name='proposal_output')([heat,box]))


def sparse_loss(target, prediction):
    import tensorflow as tf
    heat, mask = target[:,:,:,:2], target[:,:,:,2:4]
    p = tf.clip_by_value(tf.nn.sigmoid(prediction[:,:,:,:2]),1e-6,1-1e-6)
    positive = tf.cast(heat == 1.,tf.float32)*mask
    negative = tf.cast(heat < 1.,tf.float32)*mask
    focal = (-tf.math.log(p)*(1-p)**2*positive
             -tf.math.log(1-p)*p**2*(1-heat)**4*negative)
    heat_loss = tf.reduce_sum(focal)/tf.maximum(tf.reduce_sum(positive),1.)
    box_mask = target[:,:,:,8:9]
    box_loss = tf.reduce_sum(tf.abs(prediction[:,:,:,2:]-target[:,:,:,4:8])*box_mask)
    box_loss /= tf.maximum(4*tf.reduce_sum(box_mask),1.)
    return heat_loss+5*box_loss


def checked_data(root):
    root = Path(root)
    doc = read_json(root/'manifest.json')
    if (doc.get('training_approved') is not True or doc.get('test_evaluated') is not False
            or doc.get('deployment_approved') is not False or doc['grid'] != [15,20]
            or sha256(root/'targets.npy') != doc['targets_sha256']):
        raise ValueError('Require verified sparse TRAIN data')
    saved = np.load(root/'targets.npy',allow_pickle=False)
    if saved.shape != (len(doc['frames']),15,20,9) or not np.isfinite(saved).all():
        raise ValueError('Invalid sparse targets')
    images, folds, groups = [], [], []
    for row, target in zip(doc['frames'],saved,strict=True):
        training_source(row['source'])
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != row['sha256']:
            raise ValueError('Frame changed or escaped training root')
        expected, _ = encode(row['truth'],row['negative_regions'])
        if not np.array_equal(expected,target):
            raise ValueError('Stored supervision differs from reviewed geometry')
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        if pixels.shape != (240,320,3):
            raise ValueError('Unexpected training resolution')
        images.append(cv2.resize(pixels,(160,120),interpolation=cv2.INTER_AREA))
        group = source_group(Path(row['source']).name)
        groups.append(group)
        folds.append(int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5)
    folds = np.array(folds)
    fit, hold = folds != 0, folds == 0
    if not fit.any() or not hold.any():
        raise ValueError('Require both training and grouped holdout')
    for key in ('source','source_sha256'):
        if {r[key] for r,f in zip(doc['frames'],fit) if f} & {r[key] for r,f in zip(doc['frames'],hold) if f}:
            raise ValueError('Cross-fold source leakage')
    return doc,np.asarray(images,np.float32),saved,folds,groups


def train(root,seed,output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if seed not in (47,48):
        raise ValueError('Require fixed seed')
    doc,x,y,folds,groups = checked_data(root)
    fit,hold = folds!=0,folds==0
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = network()
    out.mkdir(parents=True)
    provenance = dict(manifest_sha256=sha256(Path(root)/'manifest.json'),script_sha256=sha256(__file__),
        preparation_script_sha256=doc['script_sha256'],seed=seed,epochs=80,batch_size=8,learning_rate=.001,
        parameters=model.count_params(),fixed_final_epoch=True,folds=folds.tolist(),groups=groups,
        scope='Sparse-label new localizer; known-environment grouped holdout, no independent qualification',
        test_evaluated=False,deployment_approved=False,pi_tested=False)
    write_json(out/'provenance.json',provenance)
    write_json(out/'status.json',dict(state='running'))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x[fit],y[fit])).shuffle(int(fit.sum()),seed=seed).batch(8)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(.001),loss=sparse_loss)

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite dense training')
                write_json(out/'status.json',dict(state='running',epoch=epoch+1,metrics=logs))

        model.fit(ds,epochs=80,verbose=2,callbacks=[Progress(),keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        model.save(out/'proposal.keras')
        values = model(x,training=False).numpy()
        if values.shape != (len(x),15,20,6) or not np.isfinite(values).all():
            raise ValueError('Invalid proposal outputs')
        np.save(out/'predictions.npy',values,allow_pickle=False)
        report = dict(training_loss=float(sparse_loss(tf.constant(y[fit]),tf.constant(values[fit]))),
            grouped_holdout_loss=float(sparse_loss(tf.constant(y[hold]),tf.constant(values[hold]))),
            fit_frames=int(fit.sum()),holdout_frames=int(hold.sum()),parameters=model.count_params(),
            model_sha256=sha256(out/'proposal.keras'),predictions_sha256=sha256(out/'predictions.npy'),
            provenance_sha256=sha256(out/'provenance.json'),
            scope='Loss diagnostics only; detection quality and runtime export not yet evaluated',
            test_evaluated=False,deployment_approved=False,pi_tested=False)
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(state='complete'))
        print(report)
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--seed',type=int,choices=(47,48),required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    train(a.data,a.seed,a.output)
