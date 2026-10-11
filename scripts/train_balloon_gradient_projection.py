"""Matched summed versus two-task projected gradients; identical inference graph."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_context_distillation_data import tensors, hashes
from scripts.clutter_round3_feature_data import targets, loss_function
from scripts.balloon_two_task_projection import project_tensor
from scripts.balloon_scene_constraints import decisions
from scripts.train_batch3_retraining import calculate

PROTOCOL=Path('configs/balloon-gradient-projection-20261011.json')


def train(arm,seed):
    p=read_json(PROTOCOL)
    if arm not in p['arms'] or seed not in p['seeds']:
        raise ValueError('Undeclared experiment')
    root=Path(f'runs/balloon-gradient-projection-{arm}-seed{seed}-20261011')
    if root.exists():
        raise FileExistsError(root)
    data=tensors('admit')
    settings=data['p']
    if any(p[k]!=settings[k] for k in ('epochs','batch_size','optimizer_steps','learning_rate','cosine_alpha')):
        raise ValueError('Matched training schedule changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    keras.utils.set_random_seed(seed)
    rng=np.random.default_rng(seed)
    model=keras.models.load_model(settings['parent'],compile=False)
    if model.count_params()!=7763:
        raise ValueError('Architecture changed')
    teacher=model(data['teacher_x'],training=False).numpy()
    probe_before=model(data['teacher_probes'],training=False).numpy()
    target=targets(data['y'],teacher,data['preserve'],data['scene_mask'])
    objective=loss_function(settings['scene_kl_weight'],settings['scene_centered_logit_weight'])
    schedule=keras.optimizers.schedules.CosineDecay(p['learning_rate'],p['optimizer_steps'],alpha=p['cosine_alpha'])
    optimizer=keras.optimizers.Adam(schedule)
    shapes=[v.shape for v in model.trainable_weights]
    sizes=[int(np.prod(s)) for s in shapes]

    @tf.function(reduce_retracing=True)
    def step(x,y,w):
        with tf.GradientTape(persistent=True) as tape:
            losses=objective(y,model(x,training=True))*w
            scene=y[:,7]>.5
            supervised=tf.reduce_mean(tf.where(scene,0.,losses))
            distilled=tf.reduce_mean(tf.where(scene,losses,0.))
        vectors=[]
        for value in (supervised,distilled):
            grads=tape.gradient(value,model.trainable_weights)
            vectors.append(tf.concat([tf.reshape(tf.zeros_like(v) if g is None else g,[-1])
                for g,v in zip(grads,model.trainable_weights,strict=True)],axis=0))
        del tape
        a,b=vectors
        combined=project_tensor(a,b) if arm=='project' else a+b
        tf.debugging.assert_all_finite(combined,'Nonfinite training gradients')
        chunks=tf.split(combined,sizes)
        optimizer.apply_gradients([(tf.reshape(g,s),v) for g,s,v in zip(chunks,shapes,model.trainable_weights,strict=True)])
        return supervised+distilled,tf.cast(tf.reduce_sum(a*b)<0,tf.int32)

    sources=dict(data['sources_sha256'])
    for path in (PROTOCOL,__file__,'scripts/balloon_two_task_projection.py',
                 'research/balloon-red-blue-20261008/balloon-context-gradient-decision-20261011.json'):
        sources[str(path)]=sha256(path)
    provenance=dict(arm=arm,seed=seed,settings=p,context_settings=settings,sources_sha256=sources,
        tensor_sha256=hashes(data),sample_identity=data['sample_identity'],scene_identities=data['scene_ids'],
        context_probes=data['context_probes'],context_scenes=data['context_scenes'],
        fixed_final_epoch=True,fresh_optimizer=True,parameters=7763,
        development_evaluated=False,test_evaluated=False,deployment_approved=False)
    root.mkdir(parents=True)
    write_json(root/'provenance.json',provenance)
    write_json(root/'status.json',dict(state='running'))
    try:
        trace=[]
        for epoch in range(p['epochs']):
            order=rng.permutation(len(data['x']))
            total,conflicts=0.,0
            for i in range(0,len(order),p['batch_size']):
                ids=order[i:i+p['batch_size']]
                value,conflict=step(data['x'][ids],target[ids],data['weights'][ids])
                total+=float(value)*len(ids)
                conflicts+=int(conflict)
            row=dict(epoch=epoch+1,loss=total/len(order),conflicting_batches=conflicts,
                     optimizer_steps=int(optimizer.iterations))
            if not np.isfinite(row['loss']):
                raise ValueError('Nonfinite epoch loss')
            trace.append(row)
            write_json(root/'trace.json',trace)
            write_json(root/'status.json',dict(state='running',**row))
            if (epoch+1)%10==0:
                print(dict(arm=arm,seed=seed,**row),flush=True)
        if int(optimizer.iterations)!=p['optimizer_steps']:
            raise ValueError('Incomplete optimizer schedule')
        after=model(data['x'],training=False).numpy()
        probes=model(data['probes'],training=False).numpy()
        if not np.isfinite(after).all() or not np.isfinite(probes).all():
            raise ValueError('Nonfinite saved predictions')
        report=calculate(teacher[:2074],after[:2074],data['y'][:2074],probes,data['probe_labels'])
        report['legacy_perfect_crop_gate']=report.pop('training_gate_pass')
        report['scene_changed_rows']=np.flatnonzero(decisions(teacher[-385:])!=decisions(after[-385:])).tolist()
        model.save(root/'student.keras')
        np.savez_compressed(root/'predictions.npz',before=teacher,after=after,labels=data['y'],
            probe_before=probe_before,probe_after=probes,probe_labels=data['probe_labels'])
        if any(sha256(k)!=h for k,h in sources.items()):
            raise ValueError('Training source changed')
        report.update(model_sha256=sha256(root/'student.keras'),predictions_sha256=sha256(root/'predictions.npz'),
            provenance_sha256=sha256(root/'provenance.json'),trace_sha256=sha256(root/'trace.json'),
            optimizer_steps=int(optimizer.iterations),training_gate_pass=False,
            scope='TRAIN fit only; eligibility pending separate scene audit, no qualification.')
        write_json(root/'report.json',report)
        write_json(root/'status.json',dict(state='complete'))
        print(dict(arm=arm,seed=seed,new_correct=sum(report['probe_correct']),lost=len(report['lost_old_rows'])),flush=True)
    except BaseException as exc:
        write_json(root/'status.json',dict(state='failed',error=str(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('sum','project'),required=True)
    p.add_argument('--seed',type=int,choices=(47,48),required=True)
    a=p.parse_args()
    train(a.arm,a.seed)
