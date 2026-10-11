"""Matched current-model hard-negative training; fixed final epoch only."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from dtr.runtime import softmax
from scripts.blue_object_data import inputs,REVIEW,QUEUE
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.balloon_positive_protection import retention

PROTOCOL = Path('configs/blue-object-trial-20261011.json')


def run_path(mode,seed):
    p = read_json(PROTOCOL)
    if mode not in p['arms'] or seed not in p['seeds']:
        raise ValueError('Undeclared arm/seed')
    return Path(f'runs/blue-object-trial-{mode}-seed{seed}-20261011')


def calculate(a):
    for k,n in (('after',1681),('blue_after',23),('clutter_after',22),('real_after',39),('new_after',13)):
        if a[k].shape != (n,3) or not np.isfinite(a[k]).all():
            raise ValueError('Invalid prediction shape/value')
    r = dict(retention=retention(a['before'],a['after'],a['labels'],a['ordinary']),
        blue_accepted=int(((a['blue_after'].argmax(1)==2)&(softmax(a['blue_after'])[:,2]>=.8)).sum()),
        old_clutter_correct=int((a['clutter_after'].argmax(1)==0).sum()),
        real_blue_correct=int((a['real_after'].argmax(1)==0).sum()),
        new_background_correct=int((a['new_after'].argmax(1)==0).sum()))
    r['training_gate_pass'] = (all(v['retained_fraction']==1 and v['current_argmax_correct']>=v['parent_argmax_correct']
        for v in r['retention'].values()) and r['blue_accepted']==23 and r['old_clutter_correct']==22
        and r['real_blue_correct']==39 and r['new_background_correct']==13)
    return r


def train(mode,seed):
    p,out = read_json(PROTOCOL),run_path(mode,seed)
    if out.exists():
        raise FileExistsError(out)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    rows,labels,ordinary,tensors = inputs(mode)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    model = keras.models.load_model(p['parent'],compile=False)
    if model.count_params()!=7763 or model.output_shape!=(None,3):
        raise ValueError('Architecture changed')
    names = ('','blue_','clutter_','real_','new_')
    arrays = {k+'before':model(x,training=False).numpy() for k,x in zip(names,tensors,strict=True)}
    arrays.update(labels=labels,ordinary=ordinary)
    targets = np.column_stack((np.eye(3)[labels],arrays['before'],ordinary)).astype(np.float32)
    paths = [str(PROTOCOL),str(REVIEW),str(QUEUE/'review.json'),__file__,
        'scripts/blue_object_data.py','scripts/blue_false_alarm_data.py','scripts/prepare_blue_object_review.py','scripts/acquire_blue_objects.py',
        'data/raw/openimages-blue-objects-20261011/acquisition.json',
        'data/raw/openimages-blue-objects-20261011/license-verification.json',
        'scripts/grid_clutter_data.py','scripts/balloon_all_correct_protection.py',
        'scripts/balloon_positive_protection.py','scripts/train_balloon_reviewed_clutter.py']
    provenance = dict(mode=mode,seed=seed,settings=p,parameters=7763,
        sample_identity=[{k:r[k] for k in ('source_image','sha256','label','preserve','view')} for r in rows],
        sources_sha256={s:sha256(s) for s in paths},
        tensor_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in tensors],
        optimizer_reset=True,final_epoch_only=True,test_evaluated=False,deployment_approved=False)
    out.mkdir(parents=True)
    write_json(out/'provenance.json',provenance)
    write_json(out/'status.json',dict(state='running'))
    try:
        dataset = tf.data.Dataset.from_tensor_slices((tensors[0],targets,np.where(ordinary,1.,p['hard_weight']).astype(np.float32)))
        dataset = dataset.shuffle(len(rows),seed=seed).batch(p['batch_size'])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(p['learning_rate']),
            loss=all_correct_loss(p['protection_weight'],p['protection_probability_cap'],p['parent_acceptance_threshold']))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError('Nonfinite training')
                write_json(out/'status.json',dict(state='running',epoch=epoch+1,metrics=logs))

        model.fit(dataset,epochs=p['epochs'],verbose=2,callbacks=[Progress(),keras.callbacks.CSVLogger(str(out/'epochs.csv'))])
        arrays.update({k+'after':model(x,training=False).numpy() for k,x in zip(names,tensors,strict=True)})
        report = calculate(arrays)
        model.save(out/'student.keras')
        np.savez_compressed(out/'predictions.npz',**arrays)
        report.update(model_sha256=sha256(out/'student.keras'),predictions_sha256=sha256(out/'predictions.npz'),
            provenance_sha256=sha256(out/'provenance.json'),optimizer_steps=int(model.optimizer.iterations.numpy()),
            scope='TRAIN feasibility only; not scene accuracy or Pi readiness',test_evaluated=False,deployment_approved=False)
        if any(sha256(s)!=h for s,h in provenance['sources_sha256'].items()):
            raise ValueError('Source changed while training')
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(state='complete'))
        print(dict(mode=mode,seed=seed,report=report))
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)))
        raise


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('control','admit'))
    p.add_argument('--seed',required=True,type=int,choices=(47,48))
    args = p.parse_args()
    train(args.mode,args.seed)

