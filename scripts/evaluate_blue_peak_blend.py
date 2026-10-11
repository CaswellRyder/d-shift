"""Fixed2x2 localizer/decoder ablation with exact controls and no model fitting."""
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.balloon_grid_runtime import GridProposals
from scripts.compare_current_blue_search import MODEL,MODEL_SHA
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.research_predictor import classify_candidates
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
PROTOCOL = Path('configs/blue-peak-blend-20261011.json')
REFERENCE = BASE/'blue-head-only-scenes-20261011.json'
OUTPUT = BASE/'blue-peak-blend-scenes-20261011.json'
RAW = Path('runs/blue-peak-blend-scenes-20261011.json')


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    reference = read_json(REFERENCE)
    previous = Path(reference['raw']['path'])
    if (sha256(previous) != reference['raw']['sha256'] or sha256(MODEL) != MODEL_SHA
            or reference['script_sha256'] != sha256('scripts/evaluate_blue_head_only.py')
            or reference['classifier_sha256'] != MODEL_SHA
            or reference['classifier_metadata_sha256'] != sha256(MODEL.with_suffix('.json'))
            or any(sha256(p) != h for p,h in reference['helpers_sha256'].items())
            or any(sha256(p) != h for p,h in reference['inputs'].items())):
        raise ValueError('Baseline sources changed')
    networks = {}
    for name,cls in (('parent',GridProposals),('background',DenseProposals)):
        identity = reference['models'][name]
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Localizer changed')
        networks[name] = cls(path)
    predictor = Predictor(MODEL,allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Threshold changed')
    native_path = Path('artifacts/balloon-regions-mac-20261008/regions.so')
    native = NativeBalloonRegions(native_path,direct=True)
    if read_json(native_path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    sources = [(s,rgb) for s,rgb in source_frames() if s['origin'] == 'complete_fit']
    train_pixels = {s['source']:(s['source_file_sha256'],s['truth'],rgb) for s,rgb in sources}
    if len(train_pixels) != 64:
        raise ValueError('TRAIN coverage changed')
    inputs = {str(REFERENCE):sha256(REFERENCE),str(previous):sha256(previous)}
    prior = read_json(previous)
    ordered_panels = ['training']+[p for p in prior if p != 'training']
    full,panels = {},{}
    for panel in ordered_panels:
        saved = prior[panel]
        for name in networks:
            if checked_metrics(saved[name]) != reference['panels'][panel][name]['metrics']:
                raise ValueError('Baseline counts changed')
        pixels = train_pixels
        if panel != 'training':
            sources,review = reviewed_frames('data/'+panel,'configs/'+panel+'.json')
            if review['threshold'] != .8 or review['matching_iou'] != .5:
                raise ValueError('Evaluation operating point changed')
            pixels = {s['source']:(s['sha256'],s['provisional_truth'],rgb) for s,rgb in sources}
            inputs['configs/'+panel+'.json'] = sha256('configs/'+panel+'.json')
        if len(pixels) != len(saved['parent']['frames']):
            raise ValueError('Scene coverage changed')
        rows = {f'{n}/{d}':[] for n in networks for d in ('original','blend')}
        changes = {n:0 for n in networks}
        for i,old in enumerate(saved['parent']['frames']):
            digest,truth,rgb = pixels[old['source']]
            if digest != old['frame_sha256'] or truth != old['truth']:
                raise ValueError('Scene identity changed')
            base = search(rgb,native,True)
            for name,network in networks.items():
                canonical = network.predict(rgb)
                grid = network.interpreter.get_tensor(network.output['index'])[0]
                original,blended = decode(grid,False),decode(grid,True)
                if original != canonical:
                    raise ValueError('Original decoder is not exactly equivalent')
                changes[name] += int(original != blended)
                for policy,learned in (('original',original),('blend',blended)):
                    proposed = supplement(base,learned)
                    if proposed[:len(base)] != base or len(proposed) > min(12,len(base)+2):
                        raise ValueError('Proposal budget or base changed')
                    ds = confidence_selection(classify_candidates(predictor,rgb,proposed))
                    row = dict({k:old[k] for k in ('id','source','frame_sha256','truth')},
                        detections=ds,counts=detection_counts(truth,ds),neural_calls=len(proposed))
                    if policy == 'original':
                        check_baseline(row,saved[name]['frames'][i])
                    rows[f'{name}/{policy}'].append(row)
        full[panel],panels[panel] = {},{}
        for name,values in rows.items():
            counts = {c:{k:sum(r['counts'][c][k] for r in values) for k in ('tp','fp','fn')} for c in LABELS}
            result = dict(frames=values,metrics=metrics(counts))
            checked_metrics(result)
            full[panel][name] = result
            original_name = name.split('/')[0]+'/original'
            panels[panel][name] = dict(metrics=result['metrics'],**compare_frames(values,rows['parent/original']),
                nonregression=nonregression(saved['parent']['metrics'],result['metrics']),
                against_own_original=compare_frames(values,rows[original_name]),
                meets_94=meets_target(result['metrics']),frames=len(values),
                neural_calls=sum(r['neural_calls'] for r in values),
                decoder_changed_frames=changes[name.split('/')[0]] if name.endswith('/blend') else 0)
        print(panel,{k:dict(metrics=v['metrics'],calls=v['neural_calls'],lost=len(v['lost_parent_targets']),
            changed=v['decoder_changed_frames']) for k,v in panels[panel].items()},flush=True)
    write_json(RAW,full)
    eligibility = {name:all(p[name]['nonregression'] and not p[name]['lost_parent_targets'] for p in panels.values())
        for name in ('parent/blend','background/blend')}
    helpers = dict(reference['helpers_sha256'])
    helpers['scripts/research_blue_peak_blend.py'] = sha256('scripts/research_blue_peak_blend.py')
    write_json(OUTPUT,dict(panels=panels,models=reference['models'],inputs=inputs,
        raw=dict(path=str(RAW),sha256=sha256(RAW)),script_sha256=sha256(__file__),protocol_sha256=sha256(PROTOCOL),
        helpers_sha256=helpers,eligible_for_pi_comparison=eligibility,
        scope='Fixed decoder2x2 experiment, original grids/decisions exactly reproduced. TRAIN diagnostic and related repeated development, not independent qualification. No threshold tuning or model fitting.',
        classifier_sha256=MODEL_SHA,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__ == '__main__':
    evaluate()
