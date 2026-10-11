"""Reconstruct context training and judge complete TRAIN scenes, not perfect crop fit."""
import copy
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.balloon_context_distillation_data import PROTOCOL, tensors, hashes
from scripts.clutter_round3_feature_data import hashes as original_hashes
from scripts.balloon_scene_constraints import RAW as TRAIN, RAW_SHA, decisions
from scripts.train_batch3_retraining import calculate
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT=Path('research/balloon-red-blue-20261008/balloon-context-distillation-20261011.json')
RAW=Path('runs/balloon-context-distillation-scenes-20261011.json')
CONTROL_AUDIT=Path('research/balloon-red-blue-20261008/clutter-round3-feature-20261011.json')


def scene_gate(effect, baseline_false, current_false):
    return (not effect['lost_targets'] and all(v['fp']<=0 and v['tp']>=0 for v in effect['count_delta'].values())
            and current_false<=baseline_false)


def false_acceptances(logits):
    return int(np.count_nonzero(decisions(logits)))


def check_control(prov, data, seed):
    keys=('parent','parent_sha256','seeds','epochs','batch_size','learning_rate','cosine_alpha',
          'hard_weight','parameters','new_repetitions','rows','optimizer_steps','scene_kl_weight','scene_centered_logit_weight')
    if (prov['arm']!='admit' or prov['seed']!=seed
            or any(prov['settings'][k]!=data['p'][k] for k in keys)
            or prov['tensor_sha256']!=original_hashes(data)
            or not np.array_equal(data['teacher_x'],data['x'])
            or not np.array_equal(data['teacher_probes'],data['probes'])
            or prov['scene_identities']!=data['scene_ids']
            or prov['sample_identity']!=data['sample_identity']):
        raise ValueError('Existing control does not exactly match context recipe')


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve context audit')
    p=read_json(PROTOCOL)
    prior=read_json(CONTROL_AUDIT)
    if (sha256(p['parent'])!=p['parent_sha256'] or sha256(TRAIN)!=RAW_SHA
            or prior['script_sha256']!=sha256('scripts/audit_clutter_round3_feature.py')
            or prior['data_helper_sha256']!=sha256('scripts/clutter_round3_feature_data.py')
            or prior['protocol_sha256']!=sha256('configs/clutter-round3-feature-20261011.json')):
        raise ValueError('Reference or control audit changed')
    raw=read_json(TRAIN)
    if set(raw)!={'training'}:
        raise ValueError('TRAIN-only scene evaluation')
    frames=raw['training']['parent']['frames']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent=keras.models.load_model(p['parent'],compile=False)
    results,scenes={},{}
    for arm in p['arms']:
        data=tensors(arm)
        teacher=parent(data['teacher_x'],training=False).numpy()
        probe_before=parent(data['teacher_probes'],training=False).numpy()
        original=replay(frames,softmax(teacher[-385:]))
        for a,b in zip(original['frames'],frames,strict=True):
            check_replayed_parent(a,b)
        if 'parent' in scenes and scenes['parent']!=original:
            raise ValueError('Context arms differ in original reference')
        scenes['parent']=original
        context_frames=copy.deepcopy(frames)
        i=0
        for f in context_frames:
            for d in f['detections']:
                identity=data['context_scenes'][i]
                if f['source']!=identity['source'] or d['crop_box']!=identity['original_crop']:
                    raise ValueError('Context proposal order changed')
                d['crop_box']=identity['crop_box']
                i+=1
        for seed in p['seeds']:
            root=Path(p['controls'][str(seed)]) if arm=='control' else Path(f'runs/balloon-context-distillation-admit-seed{seed}-20261011')
            prov,report=read_json(root/'provenance.json'),read_json(root/'report.json')
            if (read_json(root/'status.json')['state']!='complete'
                    or report['model_sha256']!=sha256(root/'student.keras')
                    or report['predictions_sha256']!=sha256(root/'predictions.npz')
                    or report['provenance_sha256']!=sha256(root/'provenance.json')
                    or report['optimizer_steps']!=p['optimizer_steps']
                    or any(sha256(k)!=h for k,h in prov['sources_sha256'].items())
                    or prov['fixed_final_epoch'] is not True or prov['fresh_optimizer'] is not True
                    or any(prov[k] is not False for k in ('development_evaluated','test_evaluated','deployment_approved'))):
                raise ValueError('Training artifact or scope changed')
            if arm=='control':
                check_control(prov,data,seed)
                old=prior['results'][f'admit{seed}']
                if old['run']!=str(root) or old['report_sha256']!=sha256(root/'report.json'):
                    raise ValueError('Reused control differs from existing audit')
            elif (prov['settings']!=p or prov['arm']!=arm or prov['seed']!=seed
                    or prov['tensor_sha256']!=hashes(data) or prov['sample_identity']!=data['sample_identity']
                    or prov['scene_identities']!=data['scene_ids'] or prov['context_probes']!=data['context_probes']
                    or prov['context_scenes']!=data['context_scenes']):
                raise ValueError('Context tensor or crop identity changed')
            with (root/'epochs.csv').open() as handle:
                trace=list(csv.DictReader(handle))
            if (len(trace)!=80 or [int(r['epoch']) for r in trace]!=list(range(80))
                    or not all(np.isfinite(float(r['loss'])) for r in trace)):
                raise ValueError('Incomplete training schedule')
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
            changes=np.flatnonzero(decisions(teacher[-385:])!=decisions(after[-385:])).tolist()
            legacy=measured.pop('training_gate_pass') and not bool(changes)
            if (any(report[k]!=v for k,v in measured.items()) or report['scene_changed_rows']!=changes
                    or (arm=='admit' and (report['legacy_perfect_crop_gate']!=legacy or report['training_gate_pass'] is not False))
                    or (arm=='control' and report['training_gate_pass']!=legacy)):
                raise ValueError('Crop diagnostics do not reproduce')
            result=replay(context_frames,softmax(after[-385:]))
            effect=difference(result,original)
            bfalse,afalse=false_acceptances(probe_before),false_acceptances(probes)
            gate=scene_gate(effect,bfalse,afalse)
            name=f'{arm}{seed}'
            scenes[name]=result
            results[name]=dict(run=str(root),model_sha256=sha256(root/'student.keras'),
                report_sha256=sha256(root/'report.json'),provenance_sha256=sha256(root/'provenance.json'),
                predictions_sha256=sha256(root/'predictions.npz'),epochs_sha256=sha256(root/'epochs.csv'),
                metrics=result['metrics'],effect=effect,background_false_acceptances=[bfalse,afalse],
                crop_diagnostics=measured,legacy_perfect_crop_gate=legacy,
                training_scene_gate_pass=gate,context='original' if arm=='control' else 'wider1.5_min16',
                reused_control=arm=='control')
            print(dict(name=name,metrics=result['metrics'],lost=len(effect['lost_targets']),
                background_false_acceptances=[bfalse,afalse],scene_gate=gate),flush=True)
    write_json(RAW,scenes)
    passed=all(results[f'admit{s}']['training_scene_gate_pass'] for s in p['seeds'])
    write_json(OUTPUT,dict(results=results,protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        raw=dict(path=str(RAW),sha256=sha256(RAW)),control_audit_sha256=sha256(CONTROL_AUDIT),
        eligible_for_export_and_development=passed,
        decision='Eligible for separate export parity and repeatedDEV, not promotion.' if passed else
                 'Do not promote: matched-context retraining failed TRAIN scene nonregression.',
        scope='TRAIN-only input/checkpoint reconstruction and cached-proposal scene replay. Both context seeds required. Legacy perfect13crop gate is reported but deliberately not optimized or used as qualification.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__=='__main__':
    audit()
