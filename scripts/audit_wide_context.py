"""Verify final checkpoints against pixels, frozen teacher outputs and gates."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json,sha256,write_json
from scripts.runtime_positive_weights import weights
from scripts.balloon_wide_context import build,cost
from scripts.train_wide_context import PROTOCOL,inputs,run_path,calculate

OUTPUT = Path('research/balloon-red-blue-20261008/wide-context-training-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    if sha256(p['parent'])!=p['parent_sha256']:
        raise ValueError('Parent changed')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'],compile=False)
    names = ('','blue_','clutter_','real_','new_','runtime_')
    results = {}
    if sha256(p['control_trial'])!=p['control_trial_sha256']:
        raise ValueError('Original matched trial changed')
    control_trial = read_json(p['control_trial'])
    for mode in p['arms']:
        rows,labels,ordinary,tensors,runtime_labels = inputs(mode)
        teacher = [parent(x,training=False).numpy() for x in tensors]
        identity = [{k:r[k] for k in ('source_image','sha256','label','preserve','view')} for r in rows]
        for seed in p['seeds']:
            old_report = control_trial['trials'][f'admit{seed}']
            old_run = Path(old_report['path'])
            old_prov = read_json(old_run/'provenance.json')
            if (old_prov['sample_identity']!=identity or old_prov['seed']!=seed
                    or old_prov['tensor_sha256']!=[hashlib.sha256(x.tobytes()).hexdigest() for x in tensors]
                    or any(old_prov['settings'][k]!=p[k] for k in ('parent','parent_sha256','epochs','batch_size',
                        'learning_rate','hard_weight','protection_weight','protection_probability_cap','parent_acceptance_threshold','positive_weight'))
                    or any(sha256(s)!=h for s,h in old_prov['sources_sha256'].items())):
                raise ValueError('Controls are not matched')
            for key,file in (('model_sha256','student.keras'),('predictions_sha256','predictions.npz'),('provenance_sha256','provenance.json')):
                if old_report[key]!=sha256(old_run/file):
                    raise ValueError('Control changed')
            results[f'control{seed}'] = old_report
            run = run_path(mode,seed)
            prov,report = read_json(run/'provenance.json'),read_json(run/'report.json')
            if (read_json(run/'status.json')['state']!='complete' or prov['mode']!=mode or prov['seed']!=seed
                    or prov['settings']!=p or prov['sample_identity']!=identity or prov['parameters']!=8563 or prov['architecture']!=cost() or prov['identity_initialization_verified'] is not True
                    or prov['optimizer_reset'] is not True or prov['final_epoch_only'] is not True
                    or any(sha256(s)!=h for s,h in prov['sources_sha256'].items())
                    or prov['tensor_sha256']!=[hashlib.sha256(x.tobytes()).hexdigest() for x in tensors]
                    or report['optimizer_steps']!=27360):
                raise ValueError('Training recipe/pixels/steps changed')
            for key,file in (('model_sha256','student.keras'),('predictions_sha256','predictions.npz'),('provenance_sha256','provenance.json')):
                if report[key]!=sha256(run/file):
                    raise ValueError('Output identity changed')
            with np.load(run/'predictions.npz') as saved:
                a = {k:saved[k] for k in saved.files}
            if (not np.array_equal(a['labels'],labels) or not np.array_equal(a['ordinary'],ordinary)
                    or not np.array_equal(a['runtime_labels'],runtime_labels)):
                raise ValueError('Labels changed')
            model = keras.models.load_model(run/'student.keras',compile=False)
            if model.count_params()!=8563:
                raise ValueError('Architecture changed')
            initial = build(parent)
            for prefix,x,before in zip(names,tensors,teacher,strict=True):
                if not np.array_equal(initial(x,training=False).numpy(),before):
                    raise ValueError('Initialization is not identity')
                if (not np.allclose(before,a[prefix+'before'],rtol=0,atol=1e-5)
                        or not np.allclose(model(x,training=False).numpy(),a[prefix+'after'],rtol=0,atol=1e-5)):
                    raise ValueError('Checkpoint does not reproduce saved outputs')
            sample_weights,conflicts = weights(ordinary,np.arange(len(labels))>=1685,a['before'],labels,p['hard_weight'],p['positive_weight'])
            if prov['sample_weights_sha256']!=hashlib.sha256(sample_weights.tobytes()).hexdigest():
                raise ValueError('Sample weighting changed')
            if prov['teacher_conflict_indices']!=np.flatnonzero(conflicts).tolist() or conflicts.sum()!=2:
                raise ValueError('Correction scope changed')
            fresh = calculate(a)
            if any(report[k]!=v for k,v in fresh.items()):
                raise ValueError('Recomputed gate differs')
            results[f'{mode}{seed}'] = dict(path=str(run),**report)
    eligible = all(results[f'admit{s}']['training_gate_pass'] and
        results[f'control{s}']['training_gate_pass'] for s in p['seeds'])
    write_json(OUTPUT,dict(trials=results,eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),control_trial_sha256=p['control_trial_sha256'],
        decision='Export all paired models for fixed development evaluation' if eligible else 'No export: paired training gate failed',
        scope='Repeated TRAIN crops and exact model reproduction. Not independent accuracy.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(eligible_for_development=eligible,results={k:dict(runtime=r['runtime_accepted'],new=r['new_background_correct'],
        blue=r['blue_accepted'],old=r['old_clutter_correct'],real=r['real_blue_correct'],gate=r['training_gate_pass']) for k,r in results.items()}))


if __name__=='__main__':
    audit()

