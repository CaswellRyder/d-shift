"""Integrate frozen paired localizers with lower-cost search, not new model fitting."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.balloon_grid_runtime import GridProposals
from scripts.blue_scene_localizer_data import PROTOCOL as TRAIN_PROTOCOL, run_path
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_indoor5_runtime_review import SEARCH, SEARCH_SHA
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_blue_bright_search import search
from scripts.research_predictor import classify_candidates
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
PROTOCOL = Path('configs/background-bright-integration-20261011.json')
TRAIN = BASE/'blue-scene-localizer-training-20261011.json'
HELPERS = ('scripts/research_blue_bright_search.py','scripts/research_balloon_search_fast.py',
    'scripts/research_balloon_search.py','scripts/research_balloon_confidence_selection.py',
    'scripts/balloon_dense_runtime.py','scripts/balloon_grid_runtime.py',
    'scripts/audit_balloon_dense_proposals.py','scripts/research_predictor.py',
    'scripts/prepare_blue_false_alarm_review.py','scripts/evaluate_reviewed_balloon_scenes.py',
    'scripts/audit_indoor5_outcomes.py','scripts/evaluate_red_blue_development.py',
    'scripts/summarize_public_balloon_trial.py')


def load_models():
    receipt = read_json(TRAIN)
    if (sha256(MODEL) != MODEL_SHA or sha256(SEARCH) != SEARCH_SHA
            or receipt['protocol_sha256'] != sha256(TRAIN_PROTOCOL)
            or receipt['script_sha256'] != sha256('scripts/export_blue_scene_localizer.py')):
        raise ValueError('Audited sources changed')
    networks = {'parent':GridProposals(SEARCH)}
    identities = {'parent':dict(path=str(SEARCH),sha256=SEARCH_SHA,metadata_sha256=sha256(SEARCH.with_suffix('.json')))}
    for arm in ('control','background'):
        for seed in (47,48):
            key = f'{arm}{seed}'
            run = run_path(arm,seed)
            path = run/'proposal.fp32.tflite'
            model = DenseProposals(path)
            if (model.metadata != receipt['results'][key]['metadata']
                    or model.metadata['provenance_sha256'] != sha256(run/'provenance.json')
                    or model.metadata['source_checkpoint_sha256'] != sha256(run/'proposal.keras')):
                raise ValueError('Changed paired export')
            networks[key] = model
            identities[key] = dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
    if list(networks) != read_json(PROTOCOL)['models']:
        raise ValueError('Fixed comparison arms changed')
    return networks,identities


def evaluate(stage):
    output = BASE/f'background-bright-{stage}-20261011.json'
    raw_path = Path(f'runs/background-bright-{stage}-20261011.json')
    if output.exists() or raw_path.exists():
        raise FileExistsError(output)
    reference_path = BASE/f'blue-bright-hybrid-{stage}-20261011.json'
    reference = read_json(reference_path)
    previous = Path(reference['raw']['path'])
    if (sha256(previous) != reference['raw']['sha256']
            or reference['classifier_sha256'] != MODEL_SHA
            or reference['classifier_metadata_sha256'] != sha256(MODEL.with_suffix('.json'))
            or any(sha256(p) != h for p,h in reference['helpers_sha256'].items())):
        raise ValueError('Frozen baseline changed')
    networks,identities = load_models()
    predictor = Predictor(MODEL,allow_unvalidated=True)
    native_path = Path('artifacts/balloon-regions-mac-20261008/regions.so')
    native = NativeBalloonRegions(native_path,direct=True)
    if (predictor.metadata['threshold'] != .8
            or read_json(native_path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c')):
        raise ValueError('Classifier or native source changed')
    inputs = {str(reference_path):sha256(reference_path),str(previous):sha256(previous),str(TRAIN):sha256(TRAIN)}
    if stage == 'training':
        sources = [(s,rgb) for s,rgb in source_frames() if s['origin'] == 'complete_fit']
        pixels = {s['source']:(s['source_file_sha256'],s['truth'],rgb) for s,rgb in sources}
        if len(pixels) != 64:
            raise ValueError('TRAIN scene coverage changed')
    elif stage == 'development':
        training_path = BASE/'background-bright-training-20261011.json'
        training = read_json(training_path)
        if (training['script_sha256'] != sha256(__file__) or training['protocol_sha256'] != sha256(PROTOCOL)
                or any(sha256(p) != h for p,h in training['helpers_sha256'].items())):
            raise ValueError('TRAIN screen changed')
        inputs[str(training_path)] = sha256(training_path)
    else:
        raise ValueError('Unknown stage')
    full,panels = {},{}
    for panel,saved in read_json(previous).items():
        parent = saved['blue_bright']
        if checked_metrics(parent) != reference['panels'][panel]['blue_bright']['metrics']:
            raise ValueError('Parent counts changed')
        if stage == 'development':
            sources,review = reviewed_frames('data/'+panel,'configs/'+panel+'.json')
            if review['threshold'] != .8 or review['matching_iou'] != .5:
                raise ValueError('Operating point changed')
            pixels = {s['source']:(s['sha256'],s['provisional_truth'],rgb) for s,rgb in sources}
            inputs['configs/'+panel+'.json'] = sha256('configs/'+panel+'.json')
        if len(pixels) != len(parent['frames']):
            raise ValueError('Scene coverage changed')
        rows = {k:[] for k in networks}
        for old in parent['frames']:
            digest,truth,rgb = pixels[old['source']]
            if digest != old['frame_sha256'] or truth != old['truth']:
                raise ValueError('Scene identity changed')
            base = search(rgb,native,True)
            for key,network in networks.items():
                proposed = supplement(base,network.predict(rgb))
                if proposed[:len(base)] != base or len(proposed) > min(12,len(base)+2):
                    raise ValueError('Changed base or proposal budget')
                ds = confidence_selection(classify_candidates(predictor,rgb,proposed))
                row = dict({k:old[k] for k in ('id','source','frame_sha256','truth')},
                    detections=ds,counts=detection_counts(truth,ds),neural_calls=len(proposed),
                    base_calls=len(base),extra_calls=len(proposed)-len(base))
                if key == 'parent':
                    check_baseline(row,old)
                rows[key].append(row)
        full[panel],panels[panel] = {},{}
        for key,values in rows.items():
            counts = {c:{k:sum(r['counts'][c][k] for r in values) for k in ('tp','fp','fn')} for c in LABELS}
            result = dict(frames=values,metrics=metrics(counts))
            checked_metrics(result)
            full[panel][key] = result
            panels[panel][key] = dict(metrics=result['metrics'],**compare_frames(values,rows['parent']),
                nonregression=nonregression(parent['metrics'],result['metrics']),
                meets_94=meets_target(result['metrics']),frames=len(values),
                neural_calls=sum(r['neural_calls'] for r in values))
        for seed in (47,48):
            panels[panel][f'background{seed}']['nonregression_vs_control'] = nonregression(
                full[panel][f'control{seed}']['metrics'],full[panel][f'background{seed}']['metrics'])
        print(panel,{k:dict(metrics=v['metrics'],calls=v['neural_calls'],lost=len(v['lost_parent_targets']))
            for k,v in panels[panel].items()},flush=True)
    write_json(raw_path,full)
    write_json(output,dict(panels=panels,models=identities,inputs=inputs,
        raw=dict(path=str(raw_path),sha256=sha256(raw_path)),protocol_sha256=sha256(PROTOCOL),
        script_sha256=sha256(__file__),helpers_sha256={p:sha256(p) for p in HELPERS},
        classifier_sha256=MODEL_SHA,classifier_metadata_sha256=sha256(MODEL.with_suffix('.json')),
        scope='Integration of previously evaluated localizers with lower-cost search. Complete TRAIN screen and repeated related development only. Candidate selection is development-driven, not independent qualification.',
        training_performed=False,test_evaluated=False,deployment_approved=False,qualification_achieved=False,pi_tested=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('training','development'),required=True)
    evaluate(parser.parse_args().stage)
