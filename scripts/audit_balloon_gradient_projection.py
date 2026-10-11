"""Reproduce matched optimizer checkpoints and the preregistered TRAIN scene gate."""
import copy
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.train_balloon_gradient_projection import PROTOCOL
from scripts.balloon_context_distillation_data import tensors, hashes
from scripts.balloon_scene_constraints import RAW as TRAIN, RAW_SHA, decisions
from scripts.train_batch3_retraining import calculate
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances

OUTPUT=Path('research/balloon-red-blue-20261008/balloon-gradient-projection-20261011.json')
RAW=Path('runs/balloon-gradient-projection-scenes-20261011.json')


def paired_gate(results):
    if set(results)!={'sum47','sum48','project47','project48'}:
        raise ValueError('Require both optimizer arms and seeds')
    return all(results[f'project{s}']['training_scene_gate_pass'] for s in (47,48))


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve projection audit')
    p=read_json(PROTOCOL)
    data=tensors('admit')
    settings=data['p']
    if sha256(TRAIN)!=RAW_SHA or sha256(settings['parent'])!=settings['parent_sha256']:
        raise ValueError('Reference changed')
    raw=read_json(TRAIN)
    if set(raw)!={'training'}:
        raise ValueError('TRAIN-only scene evaluation')
    frames=raw['training']['parent']['frames']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent=keras.models.load_model(settings['parent'],compile=False)
    teacher=parent(data['teacher_x'],training=False).numpy()
    probe_before=parent(data['teacher_probes'],training=False).numpy()
    original=replay(frames,softmax(teacher[-385:]))
    for a,b in zip(original['frames'],frames,strict=True):
        check_replayed_parent(a,b)
    context_frames=copy.deepcopy(frames)
    i=0
    for f in context_frames:
        for d in f['detections']:
            row=data['context_scenes'][i]
            if f['source']!=row['source'] or d['crop_box']!=row['original_crop']:
                raise ValueError('Context order changed')
            d['crop_box']=row['crop_box']
            i+=1
    results,scenes={},dict(parent=original)
    for arm in p['arms']:
        for seed in p['seeds']:
            root=Path(f'runs/balloon-gradient-projection-{arm}-seed{seed}-20261011')
            prov,report=read_json(root/'provenance.json'),read_json(root/'report.json')
            if (read_json(root/'status.json')['state']!='complete'
                    or prov['settings']!=p or prov['context_settings']!=settings
                    or prov['arm']!=arm or prov['seed']!=seed or prov['tensor_sha256']!=hashes(data)
                    or prov['sample_identity']!=data['sample_identity'] or prov['scene_identities']!=data['scene_ids']
                    or prov['context_probes']!=data['context_probes'] or prov['context_scenes']!=data['context_scenes']
                    or report['optimizer_steps']!=p['optimizer_steps']
                    or prov['fixed_final_epoch'] is not True or prov['fresh_optimizer'] is not True
                    or any(sha256(k)!=h for k,h in prov['sources_sha256'].items())
                    or any(prov[k] is not False for k in ('development_evaluated','test_evaluated','deployment_approved'))):
                raise ValueError('Training identity or scope changed')
            for key,file in (('model_sha256','student.keras'),('predictions_sha256','predictions.npz'),
                             ('provenance_sha256','provenance.json'),('trace_sha256','trace.json')):
                if sha256(root/file)!=report[key]:
                    raise ValueError('Training artifact changed')
            trace=read_json(root/'trace.json')
            if (len(trace)!=80 or [r['epoch'] for r in trace]!=list(range(1,81))
                    or [r['optimizer_steps'] for r in trace]!=[157*i for i in range(1,81)]
                    or not all(np.isfinite(r['loss']) and 0<=r['conflicting_batches']<=157 for r in trace)):
                raise ValueError('Incomplete matched training schedule')
            model=keras.models.load_model(root/'student.keras',compile=False)
            if model.count_params()!=7763 or model.get_config()!=parent.get_config():
                raise ValueError('Inference graph changed')
            after=model(data['x'],training=False).numpy()
            probes=model(data['probes'],training=False).numpy()
            with np.load(root/'predictions.npz',allow_pickle=False) as stored:
                for k,value in dict(before=teacher,after=after,labels=data['y'],probe_before=probe_before,
                                     probe_after=probes,probe_labels=data['probe_labels']).items():
                    np.testing.assert_allclose(stored[k],value,atol=1e-5,rtol=0)
            measured=calculate(teacher[:2074],after[:2074],data['y'][:2074],probes,data['probe_labels'])
            legacy=measured.pop('training_gate_pass')
            changes=np.flatnonzero(decisions(teacher[-385:])!=decisions(after[-385:])).tolist()
            if (any(report[k]!=v for k,v in measured.items()) or report['legacy_perfect_crop_gate']!=legacy
                    or report['scene_changed_rows']!=changes or report['training_gate_pass'] is not False):
                raise ValueError('Crop diagnostics changed')
            result=replay(context_frames,softmax(after[-385:]))
            effect=difference(result,original)
            bf,af=false_acceptances(probe_before),false_acceptances(probes)
            name=f'{arm}{seed}'
            scenes[name]=result
            results[name]=dict(run=str(root),report_sha256=sha256(root/'report.json'),**report,
                metrics=result['metrics'],effect=effect,background_false_acceptances=[bf,af],
                training_scene_gate_pass=scene_gate(effect,bf,af),
                conflicting_batches=sum(r['conflicting_batches'] for r in trace))
            print(dict(name=name,metrics=result['metrics'],lost=len(effect['lost_targets']),
                background_false_acceptances=[bf,af],scene_gate=results[name]['training_scene_gate_pass']),flush=True)
    write_json(RAW,scenes)
    passed=paired_gate(results)
    write_json(OUTPUT,dict(results=results,protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        raw=dict(path=str(RAW),sha256=sha256(RAW)),eligible_for_export_and_development=passed,
        decision='Eligible for separate export parity and repeatedDEV.' if passed else 'Reject: projected-gradient pair failed TRAIN scene gate.',
        scope='Four fixed80epoch models; equal data/batches/steps. Summed versus projected gradient update only. TRAIN inputs and checkpoint outputs reconstructed; no independent qualification.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__=='__main__':
    audit()
