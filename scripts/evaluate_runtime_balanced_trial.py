"""Evaluate all fixed classifiers with identical frozen lower-cost research search."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_grid_runtime import GridProposals
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.export_runtime_balanced_trial import checked_trial, TRIAL, PARENT
from scripts.prepare_indoor5_runtime_review import SEARCH, SEARCH_SHA
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_blue_bright_search import search
from scripts.research_predictor import classify_candidates
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
OUTPUT = BASE/'runtime-balanced-development-20261011.json'
RAW = Path('runs/runtime-balanced-development-20261011.json')


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve evidence')
    trial = checked_trial()
    paths = {'parent':PARENT,**{k:Path(r['path'])/'student.fp32.tflite' for k,r in trial['trials'].items()}}
    models,identities = {},{}
    for name,path in paths.items():
        meta = read_json(path.with_suffix('.json'))
        if meta['threshold'] != .8 or meta['sha256'] != sha256(path):
            raise ValueError('Model contract changed')
        if name != 'parent':
            r = trial['trials'][name]
            if (meta['source_student_sha256'] != r['model_sha256'] or meta['provenance_sha256'] != r['provenance_sha256']
                    or meta['training_trial_sha256'] != sha256(TRIAL) or meta['parity_rows'] != 2466
                    or not 0 <= meta['max_keras_tflite_delta'] <= .002 or meta['parameters'] != r['parameters']
                    or not 0 <= meta['max_checkpoint_original_group_delta'] <= 1e-5
                    or meta['export_script_sha256'] != sha256('scripts/export_runtime_balanced_trial.py')
                    or meta['deployment_approved'] is not False or meta['pi_zero_verified'] is not False):
                raise ValueError('Unverified export')
        models[name] = Predictor(path,allow_unvalidated=True)
        identities[name] = dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
    reference_path = BASE/'blue-bright-hybrid-development-20261011.json'
    reference = read_json(reference_path)
    old_raw = Path(reference['raw']['path'])
    if (reference['raw']['sha256'] != sha256(old_raw) or sha256(SEARCH) != SEARCH_SHA
            or any(sha256(p) != h for p,h in reference['helpers_sha256'].items())
            or reference['classifier_sha256'] != identities['parent']['sha256']):
        raise ValueError('Frozen pipeline changed')
    localizer = GridProposals(SEARCH)
    native_path = Path('artifacts/balloon-regions-mac-20261008/regions.so')
    if read_json(native_path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    native = NativeBalloonRegions(native_path,direct=True)
    full,panels = {},{}
    for panel,saved in read_json(old_raw).items():
        sources,review = reviewed_frames('data/'+panel,'configs/'+panel+'.json')
        old = saved['blue_bright']
        if checked_metrics(old) != reference['panels'][panel]['blue_bright']['metrics']:
            raise ValueError('Baseline counts changed')
        if review['threshold'] != .8 or review['matching_iou'] != .5:
            raise ValueError('Operating point changed')
        pixels = {r['source']:(r,rgb) for r,rgb in sources}
        if len(pixels) != len(old['frames']):
            raise ValueError('Scene coverage changed')
        rows = {name:[] for name in models}
        for previous in old['frames']:
            r,rgb = pixels[previous['source']]
            if r['sha256'] != previous['frame_sha256'] or r['provisional_truth'] != previous['truth']:
                raise ValueError('Scene identity changed')
            candidates = supplement(search(rgb,native,True),localizer.predict(rgb))
            for name,predictor in models.items():
                ds = confidence_selection(classify_candidates(predictor,rgb,candidates))
                row = dict({k:previous[k] for k in ('id','source','frame_sha256','truth')},
                    detections=ds,counts=detection_counts(previous['truth'],ds),neural_calls=len(candidates))
                if name == 'parent':
                    check_baseline(row,previous)
                rows[name].append(row)
        full[panel],panels[panel] = {},{}
        for name,values in rows.items():
            counts = {c:{k:sum(r['counts'][c][k] for r in values) for k in ('tp','fp','fn')} for c in LABELS}
            result = dict(frames=values,metrics=metrics(counts))
            checked_metrics(result)
            full[panel][name] = result
            panels[panel][name] = dict(metrics=result['metrics'],**compare_frames(values,rows['parent']),
                nonregression_vs_parent=nonregression(old['metrics'],result['metrics']),
                meets_94=meets_target(result['metrics']),neural_calls=sum(r['neural_calls'] for r in values))
        for seed in (47,48):
            panels[panel][f'admit{seed}']['nonregression_vs_control'] = nonregression(
                full[panel][f'control{seed}']['metrics'],full[panel][f'admit{seed}']['metrics'])
        print(panel,{k:v['metrics'] for k,v in panels[panel].items()},flush=True)
    write_json(RAW,full)
    helpers = ('scripts/export_runtime_balanced_trial.py','scripts/research_blue_bright_search.py',
        'scripts/research_predictor.py','scripts/research_balloon_confidence_selection.py',
        'scripts/balloon_grid_runtime.py','scripts/audit_balloon_dense_proposals.py',
        'scripts/audit_indoor5_outcomes.py','scripts/evaluate_reviewed_balloon_scenes.py',
        'scripts/evaluate_red_blue_development.py','scripts/summarize_public_balloon_trial.py')
    write_json(OUTPUT,dict(panels=panels,classifiers=identities,training_trial_sha256=sha256(TRIAL),
        reference_sha256=sha256(reference_path),reference_raw_sha256=sha256(old_raw),
        raw=dict(path=str(RAW),sha256=sha256(RAW)),script_sha256=sha256(__file__),
        helpers_sha256={p:sha256(p) for p in helpers},
        reviews_sha256={'configs/'+p+'.json':sha256('configs/'+p+'.json') for p in panels},
        localizer=dict(path=str(SEARCH),sha256=SEARCH_SHA,metadata_sha256=sha256(SEARCH.with_suffix('.json'))),
        scope='All final paired checkpoints, identical frozen proposals and operating point. Repeated related development, not independent qualification. No Pi timing or deployment claim.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))


if __name__ == '__main__':
    evaluate()
