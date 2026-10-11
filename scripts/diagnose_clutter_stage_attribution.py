"""Read-only paired stage attribution on repeated DEV; never select or promote."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.fit_clutter_feature_correction import checked_sources
from scripts.export_clutter_feature_correction import checked_trial
from scripts.evaluate_clutter_feature_correction import OUTPUT as SCENES, BASELINE
from scripts.evaluate_current_confident_containment import checked_reference
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.summarize_public_balloon_trial import checked_metrics

PROTOCOL = Path('configs/clutter-stage-attribution-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/clutter-stage-attribution-20261011.json')
RAW = Path('runs/clutter-stage-attribution-20261011.json')


def difference(result, reference):
    measured, baseline = checked_metrics(result), checked_metrics(reference)
    compared = compare_frames(result['frames'], reference['frames'])
    return dict(count_delta={c:{k:measured[c][k]-baseline[c][k] for k in ('tp','fp','fn')}
                             for c in measured},
        lost_targets=compared['lost_parent_targets'], gained_targets=compared['gained_targets'])


def crops(frames, pixels):
    if set(pixels) != {f['source'] for f in frames}:
        raise ValueError('Frame coverage changed')
    rows=[]
    for f in frames:
        digest, truth, rgb = pixels[f['source']]
        if digest!=f['frame_sha256'] or truth!=f['truth'] or rgb.shape!=(240,320,3):
            raise ValueError('Source identity, geometry or annotations changed')
        for d in f['detections']:
            a,b,c,e=d['crop_box']
            if not 0<=a<c<=320 or not 0<=b<e<=240:
                raise ValueError('Invalid crop')
            rows.append(np.asarray(Image.fromarray(rgb[b:e,a:c]).resize((64,64),Image.Resampling.BILINEAR),np.float32))
    return np.stack(rows)


def diagnose():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve attribution receipts')
    p=read_json(PROTOCOL)
    source, feature_audit=checked_sources()
    correction_audit,names=checked_trial()
    checked_reference()
    doc, base=read_json(SCENES),read_json(BASELINE)
    if (sha256(doc['raw']['path'])!=doc['raw']['sha256']
            or doc['script_sha256']!=sha256('scripts/evaluate_clutter_feature_correction.py')
            or doc['baseline_sha256']!=sha256(BASELINE)
            or sha256(base['raw']['path'])!=base['raw']['sha256']
            or any(sha256(k)!=h for k,h in {**doc['helpers_sha256'],**doc['inputs']}.items())
            or doc['qualification_achieved'] is not False or doc['test_evaluated'] is not False):
        raise ValueError('Frozen comparison changed')
    previous=read_json(doc['raw']['path'])
    if set(previous)!=set(p['panels']):
        raise ValueError('Unexpected panels; never read reserved test')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    paths={'parent':Path(source['reference'])}
    paths.update({'feature/'+n:Path(feature_audit['results'][n]['run'])/'student.keras' for n in names})
    paths.update({'corrected/'+n:Path(correction_audit['results'][n]['run'])/'student.keras' for n in names})
    if set(paths)!=set(p['variants']):
        raise ValueError('Incomplete paired attribution')
    models={n:keras.models.load_model(path,compile=False) for n,path in paths.items()}
    if any(m.count_params()!=7763 for m in models.values()):
        raise ValueError('Architecture changed')
    identities={n:dict(path=str(path),sha256=sha256(path)) for n,path in paths.items()}
    full,summaries,inputs={},{},{}
    for panel in p['panels']:
        frames=previous[panel]['parent']['frames']
        if panel=='training':
            pixels={s['source']:(s['source_file_sha256'],s['truth'],rgb)
                    for s,rgb in source_frames() if s['origin']=='complete_fit'}
        else:
            review_path='configs/'+panel+'.json'
            sources,review=reviewed_frames('data/'+panel,review_path)
            if review['threshold']!=.8 or review['matching_iou']!=.5:
                raise ValueError('Threshold changed')
            pixels={s['source']:(s['sha256'],s['provisional_truth'],rgb) for s,rgb in sources}
            inputs[review_path]=sha256(review_path)
        x=crops(frames,pixels)
        full[panel]={}
        for name,model in models.items():
            logits=np.concatenate([model(x[i:i+64],training=False).numpy() for i in range(0,len(x),64)])
            result=replay(frames,softmax(logits))
            if name=='parent' or name.startswith('corrected/'):
                old=previous[panel]['parent' if name=='parent' else name.split('/')[1]]
                for a,b in zip(result['frames'],old['frames'],strict=True):
                    check_replayed_parent(a,b)
            full[panel][name]=result
        summaries[panel]={}
        for name,result in full[panel].items():
            row=dict(metrics=result['metrics'],against_parent=difference(result,full[panel]['parent']))
            if name.startswith('corrected/'):
                row['correction_effect']=difference(result,full[panel]['feature/'+name.split('/')[1]])
            if 'admit' in name:
                row['new_data_effect']=difference(result,full[panel][name.replace('admit','control')])
            summaries[panel][name]=row
        print(dict(panel=panel,metrics={n:r['metrics'] for n,r in full[panel].items()}),flush=True)
    checked_sources()
    checked_trial()
    if any(sha256(r['path'])!=r['sha256'] for r in identities.values()):
        raise ValueError('Checkpoint changed during attribution')
    write_json(RAW,full)
    write_json(OUTPUT,dict(protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        scene_receipt_sha256=sha256(SCENES),models=identities,inputs=inputs,panels=summaries,
        raw=dict(path=str(RAW),sha256=sha256(RAW)),
        helpers_sha256={k:sha256(k) for k in ('scripts/diagnose_clutter_round3_feature.py',
            'scripts/evaluate_reviewed_balloon_scenes.py','scripts/audit_indoor5_outcomes.py',
            'scripts/research_confident_containment.py','scripts/summarize_public_balloon_trial.py')},
        scope=p['scope'],training_performed=False,development_evaluated=True,test_evaluated=False,
        pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__=='__main__':
    diagnose()
