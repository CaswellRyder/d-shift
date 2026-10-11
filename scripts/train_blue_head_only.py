"""Fit only the blue1x1 proposal-scoring head on frozen features and geometry."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.blue_head_objective import Objective,solve
from scripts.blue_scene_localizer_data import data,dependencies,PROTOCOL as SOURCE_PROTOCOL

PROTOCOL = Path('configs/blue-head-only-20261011.json')


def run_path(arm):
    if arm not in ('control','background'):
        raise ValueError('Unknown arm')
    return Path(f'runs/blue-head-only-{arm}-20261011')


def features_and_targets(parent,x,folds):
    import keras
    if (parent.count_params() != 2342 or parent.output_shape != (None,15,20,6)
            or len(x) != 137 or len(folds) != 137 or (folds == 0).sum() != 16):
        raise ValueError('Frozen architecture or split changed')
    features = keras.Model(parent.inputs,parent.get_layer('center_logits').input)(x,training=False).numpy()
    predictions = parent(x,training=False).numpy()
    if features.shape != (137,15,20,24) or not np.isfinite(features).all():
        raise ValueError('Unexpected frozen features')
    fit = folds[:129] != 0
    if fit.sum() != 113 or not np.all(folds[129:] != 0):
        raise ValueError('Original held-out frames entered training')
    old = features[:129][fit].reshape(-1,24).astype(np.float64)
    mean,scale = old.mean(0),np.maximum(old.std(0),1e-6)

    def design(values):
        a = (values.reshape(-1,24).astype(np.float64)-mean)/scale
        return np.column_stack((a,np.ones(len(a))))

    return features,predictions,mean,scale,design(old),design(features[129:]),design(features),fit


def train():
    p = read_json(PROTOCOL)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    if any(run_path(arm).exists() for arm in p['arms']):
        raise FileExistsError('Preserve existing trial')
    x,_,folds,identities,groups = data('background')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'],compile=False)
    features,before,mean,scale,old,new,all_design,fit = features_and_targets(parent,x,folds)
    old_logits = before[:129,...,1][fit].ravel().astype(np.float64)
    weights = np.where(np.exp(-np.logaddexp(0.,-old_logits)) >= .1,
        p['preservation_weight_above_proposal_threshold'],1.)
    negative_logits = before[129:,...,1].ravel().astype(np.float64)
    sources = {**dependencies(),str(SOURCE_PROTOCOL):sha256(SOURCE_PROTOCOL),
        str(PROTOCOL):sha256(PROTOCOL),__file__:sha256(__file__),
        'scripts/blue_head_objective.py':sha256('scripts/blue_head_objective.py')}
    for arm in p['arms']:
        root = run_path(arm)
        root.mkdir(parents=True)
        write_json(root/'status.json',dict(state='running'))
        try:
            negative_weight = p['negative_weight'] if arm == 'background' else 0.
            objective = Objective(old,weights,new,negative_logits,negative_weight,p['ridge'])
            delta,trace = solve(objective,p['max_newton_iterations'],p['gradient_tolerance'])
            model = keras.models.load_model(p['parent'],compile=False)
            layer = model.get_layer('center_logits')
            kernel,bias = layer.get_weights()
            if kernel.shape != (1,1,24,2) or bias.shape != (2,):
                raise ValueError('Scoring head changed')
            kernel[0,0,:,1] += (delta[:24]/scale).astype(np.float32)
            bias[1] += np.float32(delta[24]-np.sum(delta[:24]*mean/scale))
            layer.set_weights([kernel,bias])
            frozen_differences = []
            for old_layer,new_layer in zip(parent.layers,model.layers,strict=True):
                for i,(a,b) in enumerate(zip(old_layer.get_weights(),new_layer.get_weights(),strict=True)):
                    if old_layer.name == 'center_logits':
                        a,b = a[...,0],b[...,0]
                    if not np.array_equal(a,b):
                        frozen_differences.append((old_layer.name,i))
            if frozen_differences:
                raise ValueError('Frozen red/backbone/geometry weights changed')
            after = model(x,training=False).numpy()
            fixed_delta = float(np.max(np.abs(after[...,[0,2,3,4,5]]-before[...,[0,2,3,4,5]])))
            linear_delta = float(np.max(np.abs(after[...,1].ravel()-(before[...,1].ravel()+all_design@delta))))
            if fixed_delta != 0 or linear_delta > 2e-5 or not np.isfinite(after).all():
                raise ValueError('Blue head folding or frozen outputs failed')
            model.save(root/'proposal.keras')
            np.save(root/'predictions.npy',after,allow_pickle=False)
            np.savez_compressed(root/'optimization.npz',old=old,weights=weights,new=new,
                negative_logits=negative_logits,mean=mean,scale=scale,delta=delta)
            prov = dict(arm=arm,settings=p,sources_sha256=sources,folds=folds.tolist(),
                additional_frames=identities,original_groups=groups,
                input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                feature_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                original_fit_frames=113,negative_fit_frames=8,holdout_frames=16,
                negative_weight=negative_weight,total_parameters=2342,updated_parameters=25,
                frozen_weight_checks_passed=True,frozen_output_max_delta=fixed_delta,
                linear_fold_max_delta=linear_delta,optimizer_trace=trace,
                optimization_sha256=sha256(root/'optimization.npz'),model_sha256=sha256(root/'proposal.keras'),
                predictions_sha256=sha256(root/'predictions.npy'),
                test_evaluated=False,deployment_approved=False,qualification_achieved=False)
            if any(sha256(path) != h for path,h in sources.items()):
                raise ValueError('Sources changed during fitting')
            write_json(root/'provenance.json',prov)
            write_json(root/'status.json',dict(state='complete'))
            print(dict(arm=arm,iterations=len(trace)-1,initial_loss=trace[0]['loss'],
                final_loss=trace[-1]['loss'],gradient=trace[-1]['gradient_inf'],
                fixed_delta=fixed_delta,fold_delta=linear_delta),flush=True)
        except BaseException as exc:
            write_json(root/'status.json',dict(state='failed',error=str(exc)))
            raise


if __name__ == '__main__':
    train()
