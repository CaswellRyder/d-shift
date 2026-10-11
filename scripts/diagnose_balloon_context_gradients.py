"""Read-only TRAIN objective/gradient decomposition at frozen checkpoints."""
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_context_distillation_data import tensors, hashes, PROTOCOL
from scripts.clutter_round3_feature_data import targets, loss_function
from scripts.audit_balloon_context_distillation import OUTPUT as AUDIT, check_control
from scripts.balloon_gradient_summary import summarize

OUTPUT=Path('research/balloon-red-blue-20261008/balloon-context-gradients-20261011.json')
ARCHIVE=Path('runs/balloon-context-gradients-20261011.npz')
GROUPS={'old_crops':(0,2074),'new_negatives':(2074,2126),'scene_distillation':(2126,2511)}


def diagnose():
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError('Preserve gradient diagnostic')
    p,doc=read_json(PROTOCOL),read_json(AUDIT)
    if (doc['protocol_sha256']!=sha256(PROTOCOL)
            or doc['script_sha256']!=sha256('scripts/audit_balloon_context_distillation.py')
            or sha256(p['parent'])!=p['parent_sha256']
            or any(doc[k] is not False for k in ('development_evaluated','test_evaluated','pi_tested',
                                                'deployment_approved','qualification_achieved'))):
        raise ValueError('Context evidence changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent=keras.models.load_model(p['parent'],compile=False)
    loss=loss_function(p['scene_kl_weight'],p['scene_centered_logit_weight'])
    results,archive,identities={}, {}, {}
    for arm in ('control','admit'):
        data=tensors(arm)
        teacher=parent(data['teacher_x'],training=False).numpy()
        target=targets(data['y'],teacher,data['preserve'],data['scene_mask'])
        paths={'parent':Path(p['parent'])}
        paths.update({str(seed):Path(doc['results'][f'{arm}{seed}']['run'])/'student.keras' for seed in (47,48)})
        for name,path in paths.items():
            if name!='parent':
                row=doc['results'][f'{arm}{name}']
                root=path.parent
                prov=read_json(root/'provenance.json')
                if (sha256(path)!=row['model_sha256'] or sha256(root/'provenance.json')!=row['provenance_sha256']
                        or any(sha256(k)!=h for k,h in prov['sources_sha256'].items())):
                    raise ValueError('Checkpoint or source changed')
                if arm=='control':
                    check_control(prov,data,int(name))
                elif prov['tensor_sha256']!=hashes(data):
                    raise ValueError('Context inputs changed')
            model=keras.models.load_model(path,compile=False)
            original=[v.numpy().copy() for v in model.weights]
            if model.count_params()!=7763:
                raise ValueError('Architecture changed')

            @tf.function(reduce_retracing=True)
            def batch_gradient(x,y,w):
                with tf.GradientTape() as tape:
                    values=loss(y,model(x,training=False))*w
                    total=tf.reduce_sum(values)
                grads=tape.gradient(total,model.trainable_weights)
                return values,[tf.zeros_like(v) if g is None else g for g,v in zip(grads,model.trainable_weights,strict=True)]

            gradients,counts,group_losses={},{},{}
            key=f'{arm}/{name}'
            for group,(start,end) in GROUPS.items():
                accumulated=np.zeros(7763,float)
                values=[]
                for i in range(start,end,64):
                    stop=min(end,i+64)
                    losses,grad=batch_gradient(data['x'][i:stop],target[i:stop],data['weights'][i:stop])
                    losses=losses.numpy()
                    flat=np.concatenate([g.numpy().ravel() for g in grad]).astype(float)
                    if not np.isfinite(losses).all() or not np.isfinite(flat).all():
                        raise ValueError('Nonfinite objective or gradient')
                    accumulated+=flat
                    values.extend(losses.tolist())
                counts[group]=end-start
                gradients[group]=accumulated/(end-start)
                group_losses[group]=dict(rows=end-start,weighted_mean_loss=float(np.mean(values)),
                    contribution_to_total=float(np.sum(values)/len(data['x'])))
                archive[key+'/'+group+'/gradient']=gradients[group]
                archive[key+'/'+group+'/losses']=np.asarray(values)
            if not all(np.array_equal(a,b.numpy()) for a,b in zip(original,model.weights,strict=True)):
                raise ValueError('Read-only gradient diagnostic mutated weights')
            result=dict(groups=group_losses,gradients=summarize(gradients,counts),
                objective=sum(r['contribution_to_total'] for r in group_losses.values()))
            if name!='parent':
                with (path.parent/'epochs.csv').open() as handle:
                    trace=list(csv.DictReader(handle))
                if sha256(path.parent/'epochs.csv')!=doc['results'][f'{arm}{name}']['epochs_sha256']:
                    raise ValueError('Training trace changed')
                result['last_training_epochs']=[dict(epoch=int(r['epoch'])+1,loss=float(r['loss'])) for r in trace[-10:]]
            results[key]=result
            identities[key]=dict(path=str(path),sha256=sha256(path),tensor_sha256=hashes(data))
            print(dict(name=key,objective=result['objective'],groups=group_losses,
                gradient_summary=result['gradients']),flush=True)
    np.savez_compressed(ARCHIVE,**archive)
    write_json(OUTPUT,dict(results=results,models=identities,audit_sha256=sha256(AUDIT),
        protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        helpers_sha256={s:sha256(s) for s in ('scripts/balloon_gradient_summary.py',
            'scripts/balloon_context_distillation_data.py','scripts/clutter_round3_feature_data.py')},
        archive=dict(path=str(ARCHIVE),sha256=sha256(ARCHIVE)),
        scope='Read-only TRAIN gradients and loss at original/final checkpoints. Negative cosine demonstrates local objective conflict, not an impossibility proof or evidence that further training cannot help. Epoch-loss traces are online averages, not directly identical to frozen-checkpoint objective.',
        weights_changed=False,training_performed=False,development_evaluated=False,test_evaluated=False,
        pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__=='__main__':
    diagnose()
