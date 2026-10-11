"""Matched compact versus function-preserving dense context; fixed final epoch only."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from dtr.runtime import softmax
from scripts.dense_context_trial_data import inputs,parameters
from scripts.balloon_dense_context_lift import build
from scripts.blue_object_data import REVIEW,QUEUE
from scripts.train_blue_view_recovery import calculate as base_calculate
from scripts.balloon_all_correct_protection import all_correct_loss
from scripts.blue_photo_long_schedule import learning_rate

PROTOCOL = Path('configs/dense-context-trial-20261011.json')


def run_path(mode,seed):
    p = read_json(PROTOCOL)
    if mode not in p['arms'] or seed not in p['seeds']:
        raise ValueError('Undeclared arm/seed')
    return Path(f'runs/dense-context-trial-{mode}-seed{seed}-20261011')


def calculate(a):
    base=dict(a)
    for key in ('before','after','labels','ordinary'):
        base[key]=a[key][:1750]
    r=base_calculate(base)
    values=a['photo_after']
    if values.shape!=(7,3) or not np.isfinite(values).all():
        raise ValueError('Invalid positive-view probes')
    r['photo_blue_accepted']=int(((values.argmax(1)==2)&(softmax(values)[:,2]>=.8)).sum())
    r['training_gate_pass']=r['training_gate_pass'] and r['photo_blue_accepted']==7
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
    names = ('','blue_','clutter_','real_','new_','runtime_','photo_')
    arrays = {k+'before':model(x,training=False).numpy() for k,x in zip(names,tensors,strict=True)}
    if not ((arrays['runtime_before'].argmax(1)==2)&(softmax(arrays['runtime_before'])[:,2]>=.8)).all():
        raise ValueError('Frozen teacher must accept all69 reviewed blue views')
    if mode=='admit':
        model=build(model)
    initial_delta=0.
    for k,x in zip(names,tensors,strict=True):
        actual=model(x,training=False).numpy()
        before=arrays[k+'before']
        initial_delta=max(initial_delta,float(np.max(np.abs(actual-before))))
        if (not np.isfinite(actual).all() or initial_delta>.0001
                or not np.array_equal(actual.argmax(1),before.argmax(1))
                or not np.array_equal(softmax(actual).max(1)>=.8,softmax(before).max(1)>=.8)):
            raise ValueError('Architecture changed initial decisions')
    if model.count_params()!=parameters(mode):
        raise ValueError('Unexpected lifted architecture')
    # Construction must not change either arm's seeded minibatch sequence.
    keras.utils.set_random_seed(seed)
    arrays.update(labels=labels,ordinary=ordinary)
    targets = np.column_stack((np.eye(3)[labels],arrays['before'],ordinary)).astype(np.float32)
    paths = [str(PROTOCOL),str(REVIEW),str(QUEUE/'review.json'),__file__,
        'scripts/dense_context_trial_data.py','scripts/balloon_dense_context_lift.py',
        'research/balloon-red-blue-20261008/dense-context-lift-initialization-20261011.json',
        'scripts/blue_photo_expansion_data.py','scripts/blue_photo_long_schedule.py','scripts/blue_view_recovery_data.py','scripts/train_blue_view_recovery.py',
        'scripts/blue_balloon_candidate_data.py','configs/blue-balloon-candidates-review-20261011.json',
        'data/blue-balloon-candidates-review-20261011/review.json',
        'research/balloon-red-blue-20261008/blue-balloon-admission-20261011.json',
        'data/raw/openimages-blue-balloon-candidates-20261011/acquisition.json',
        'data/raw/openimages-blue-balloon-candidates-20261011/license-verification.json','scripts/runtime_positive_data.py','configs/runtime-positive-review-20261011.json',
        'data/runtime-positive-coverage-20261011/review.json','scripts/train_blue_objects.py',
        'scripts/blue_object_data.py','scripts/blue_false_alarm_data.py','scripts/prepare_blue_object_review.py','scripts/acquire_blue_objects.py',
        'data/raw/openimages-blue-objects-20261011/acquisition.json',
        'data/raw/openimages-blue-objects-20261011/license-verification.json',
        'scripts/grid_clutter_data.py','scripts/balloon_all_correct_protection.py',
        'scripts/balloon_positive_protection.py','scripts/train_balloon_reviewed_clutter.py']
    provenance = dict(mode=mode,seed=seed,settings=p,parameters=parameters(mode),initial_max_logit_delta=initial_delta,
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
        model.compile(optimizer=keras.optimizers.Adam(learning_rate(p)),
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
        report.update(parameters=parameters(mode),model_sha256=sha256(out/'student.keras'),predictions_sha256=sha256(out/'predictions.npz'),
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
