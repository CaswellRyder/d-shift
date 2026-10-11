"""Frozen head-only comparison on TRAIN then repeated development; no threshold tuning."""
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.balloon_grid_runtime import GridProposals
from scripts.compare_current_blue_search import MODEL,MODEL_SHA
from scripts.evaluate_background_bright import HELPERS
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.export_blue_head_only import OUTPUT as TRAIN,PROTOCOL
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_indoor5_runtime_review import SEARCH,SEARCH_SHA
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_blue_bright_search import search
from scripts.research_predictor import classify_candidates
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
OUTPUT = BASE/'blue-head-only-scenes-20261011.json'
RAW = Path('runs/blue-head-only-scenes-20261011.json')


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    trial = read_json(TRAIN)
    if (sha256(MODEL) != MODEL_SHA or sha256(SEARCH) != SEARCH_SHA
            or trial['protocol_sha256'] != sha256(PROTOCOL)
            or trial['script_sha256'] != sha256('scripts/export_blue_head_only.py')
            or trial['results']['control']['control_parent_exact'] is not True):
        raise ValueError('Verified training changed')
    networks = {'parent':GridProposals(SEARCH)}
    identities = {'parent':dict(path=str(SEARCH),sha256=SEARCH_SHA,metadata_sha256=sha256(SEARCH.with_suffix('.json')))}
    for arm in ('control','background'):
        result = trial['results'][arm]
        path = Path(result['path'])
        network = DenseProposals(path)
        prov = read_json(path.parent/'provenance.json')
        if (network.metadata != result['metadata'] or any(sha256(p) != h for p,h in prov['sources_sha256'].items())
                or network.metadata['provenance_sha256'] != sha256(path.parent/'provenance.json')
                or network.metadata['source_checkpoint_sha256'] != sha256(path.parent/'proposal.keras')):
            raise ValueError('Head-only export changed')
        networks[arm] = network
        identities[arm] = dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
    predictor = Predictor(MODEL,allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Operating point changed')
    native_path = Path('artifacts/balloon-regions-mac-20261008/regions.so')
    native = NativeBalloonRegions(native_path,direct=True)
    if read_json(native_path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    sources = [(s,rgb) for s,rgb in source_frames() if s['origin'] == 'complete_fit']
    train_pixels = {s['source']:(s['source_file_sha256'],s['truth'],rgb) for s,rgb in sources}
    if len(train_pixels) != 64:
        raise ValueError('TRAIN coverage changed')
    inputs,panels,full = {str(TRAIN):sha256(TRAIN)},{},{}
    for stage in ('training','development'):
        reference_path = BASE/f'blue-bright-hybrid-{stage}-20261011.json'
        reference = read_json(reference_path)
        previous = Path(reference['raw']['path'])
        if (sha256(previous) != reference['raw']['sha256']
                or reference['classifier_sha256'] != MODEL_SHA
                or reference['classifier_metadata_sha256'] != sha256(MODEL.with_suffix('.json'))
                or any(sha256(p) != h for p,h in reference['helpers_sha256'].items())):
            raise ValueError('Frozen baseline changed')
        inputs[str(reference_path)],inputs[str(previous)] = sha256(reference_path),sha256(previous)
        for panel,saved in read_json(previous).items():
            parent = saved['blue_bright']
            if checked_metrics(parent) != reference['panels'][panel]['blue_bright']['metrics']:
                raise ValueError('Original counts changed')
            pixels = train_pixels
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
                        raise ValueError('Proposal budget changed')
                    ds = confidence_selection(classify_candidates(predictor,rgb,proposed))
                    row = dict({k:old[k] for k in ('id','source','frame_sha256','truth')},
                        detections=ds,counts=detection_counts(truth,ds),neural_calls=len(proposed))
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
            if (panels[panel]['control']['metrics'] != panels[panel]['parent']['metrics']
                    or panels[panel]['control']['lost_parent_targets'] or panels[panel]['control']['gained_targets']):
                raise ValueError('Control no longer reproduces original targets')
            print(panel,{k:dict(metrics=v['metrics'],calls=v['neural_calls'],lost=len(v['lost_parent_targets']))
                for k,v in panels[panel].items()},flush=True)
    write_json(RAW,full)
    eligible = all(p['background']['nonregression'] and not p['background']['lost_parent_targets'] for p in panels.values())
    write_json(OUTPUT,dict(panels=panels,models=identities,inputs=inputs,
        raw=dict(path=str(RAW),sha256=sha256(RAW)),protocol_sha256=sha256(PROTOCOL),
        script_sha256=sha256(__file__),helpers_sha256={p:sha256(p) for p in HELPERS},
        classifier_sha256=MODEL_SHA,classifier_metadata_sha256=sha256(MODEL.with_suffix('.json')),
        eligible_for_pi_comparison=eligible,
        scope='TRAIN-only head fit, unchanged classifier/geometry/architecture. Repeated related development is not independent qualification. No tuning after these results.',
        test_evaluated=False,deployment_approved=False,qualification_achieved=False,pi_tested=False))


if __name__ == '__main__':
    evaluate()
