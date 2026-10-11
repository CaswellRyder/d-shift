"""Rebuild frozen-feature objective and verify every unchanged weight before export."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_dense_proposals import decode
from scripts.blue_head_objective import Objective,solve
from scripts.blue_scene_localizer_data import data
from scripts.export_balloon_dense_proposals import proposal_parity
from scripts.train_blue_head_only import PROTOCOL,run_path,features_and_targets

OUTPUT = Path('research/balloon-red-blue-20261008/blue-head-only-training-20261011.json')


def export():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    x,_,folds,identities,groups = data('background')
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    parent = keras.models.load_model(p['parent'],compile=False)
    features,before,mean,scale,old,new,all_design,fit = features_and_targets(parent,x,folds)
    old_logits = before[:129,...,1][fit].ravel().astype(np.float64)
    weights = np.where(np.exp(-np.logaddexp(0.,-old_logits)) >= .1,
        p['preservation_weight_above_proposal_threshold'],1.)
    logits = before[129:,...,1].ravel().astype(np.float64)
    results = {}
    for arm in p['arms']:
        root = run_path(arm)
        path = root/'proposal.fp32.tflite'
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
        prov = read_json(root/'provenance.json')
        if (read_json(root/'status.json')['state'] != 'complete' or prov['arm'] != arm or prov['settings'] != p
                or prov['folds'] != folds.tolist() or prov['additional_frames'] != identities or prov['original_groups'] != groups
                or prov['input_sha256'] != hashlib.sha256(x.tobytes()).hexdigest()
                or prov['feature_sha256'] != hashlib.sha256(features.tobytes()).hexdigest()
                or any(sha256(s) != h for s,h in prov['sources_sha256'].items())):
            raise ValueError('Frozen training inputs changed')
        for field,file in (('model_sha256','proposal.keras'),('predictions_sha256','predictions.npy'),
                ('optimization_sha256','optimization.npz')):
            if sha256(root/file) != prov[field]:
                raise ValueError('Training output changed')
        negative_weight = p['negative_weight'] if arm == 'background' else 0.
        objective = Objective(old,weights,new,logits,negative_weight,p['ridge'])
        delta,trace = solve(objective,p['max_newton_iterations'],p['gradient_tolerance'])
        with np.load(root/'optimization.npz') as stored:
            for name,value in dict(old=old,weights=weights,new=new,negative_logits=logits,mean=mean,scale=scale,delta=delta).items():
                if not np.array_equal(stored[name],value):
                    raise ValueError('Optimization tensors or deterministic solution changed')
        if trace != prov['optimizer_trace'] or prov['negative_weight'] != negative_weight:
            raise ValueError('Optimization did not reproduce')
        model = keras.models.load_model(root/'proposal.keras',compile=False)
        if model.count_params() != 2342:
            raise ValueError('Architecture grew')
        for a,b in zip(parent.layers,model.layers,strict=True):
            for u,v in zip(a.get_weights(),b.get_weights(),strict=True):
                if a.name == 'center_logits':
                    u,v = u[...,0],v[...,0]
                if not np.array_equal(u,v):
                    raise ValueError('Frozen weights changed')
        expected_kernel,expected_bias = parent.get_layer('center_logits').get_weights()
        expected_kernel[0,0,:,1] += (delta[:24]/scale).astype(np.float32)
        expected_bias[1] += np.float32(delta[24]-np.sum(delta[:24]*mean/scale))
        for expected,actual in zip((expected_kernel,expected_bias),model.get_layer('center_logits').get_weights(),strict=True):
            if not np.array_equal(expected,actual):
                raise ValueError('Saved blue head does not match optimized solution')
        values = model(x,training=False).numpy()
        fixed_delta = float(np.max(np.abs(values[...,[0,2,3,4,5]]-before[...,[0,2,3,4,5]])))
        if (not np.array_equal(values,np.load(root/'predictions.npy',allow_pickle=False)) or fixed_delta != 0
                or np.max(np.abs(values[...,1].ravel()-before[...,1].ravel()-all_design@delta)) > 2e-5):
            raise ValueError('Checkpoint output reconstruction failed')

        @tf.function(input_signature=[tf.TensorSpec([1,120,160,3],tf.float32)])
        def serve(image):
            return model(image,training=False)

        frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        binary = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=binary,num_threads=1)
        interpreter.allocate_tensors()
        inp,out = interpreter.get_input_details()[0],interpreter.get_output_details()[0]
        maximum = 0.
        for image,reference in zip(x,values,strict=True):
            interpreter.set_tensor(inp['index'],image[None])
            interpreter.invoke()
            actual = interpreter.get_tensor(out['index'])[0]
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite export')
            maximum = max(maximum,float(np.abs(actual-reference).max()))
            proposal_parity(decode(actual),decode(reference))
        if maximum > .002:
            raise ValueError('Export parity failed')
        path.write_bytes(binary)
        metadata = dict(kind='balloon_dense_proposal_research',experiment='blue_head_only',arm=arm,
            sha256=sha256(path),bytes=len(binary),source_checkpoint_sha256=prov['model_sha256'],
            provenance_sha256=sha256(root/'provenance.json'),protocol_sha256=sha256(PROTOCOL),parameters=2342,
            input_shape=[1,120,160,3],output_shape=[1,15,20,6],tensor_dtype='float32',
            proposal_threshold=.1,max_proposals=12,max_added_blue_proposals=2,
            decoder_sha256=sha256('scripts/audit_balloon_dense_proposals.py'),export_script_sha256=sha256(__file__),
            preprocessing='320x240 RGB uint8 to160x120 INTER_AREA float32 range0..255',
            max_tensor_delta=maximum,decoded_parity_frames=len(x),test_evaluated=False,deployment_approved=False,pi_tested=False)
        write_json(path.with_suffix('.json'),metadata)
        results[arm] = dict(path=str(path),metadata=metadata,optimizer_trace=trace,
            frozen_output_max_delta=fixed_delta,control_parent_exact=bool(np.array_equal(values,before)),
            parent_negative_proposals=[len(decode(v)) for v in before[-8:]],
            candidate_negative_proposals=[len(decode(v)) for v in values[-8:]])
        print(arm,{k:v for k,v in results[arm].items() if k not in ('metadata','optimizer_trace')},flush=True)
    write_json(OUTPUT,dict(results=results,protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        scope='Reconstructed TRAIN-only25-parameter convex fit and export parity. No scene accuracy selection or independent qualification.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__ == '__main__':
    export()
