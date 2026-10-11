"""Frozen paired classifiers with unchanged MSER/coarse47 development searches."""
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from dtr.runtime import Predictor
from scripts.export_blue_view_recovery import checked_trial,TRIAL,PARENT
from scripts.balloon_grid_runtime import GridProposals
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_balloon_grid_clutter import nonregression
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.audit_current_blue_search import matched_truth

OUTPUT = Path('research/balloon-red-blue-20261008/blue-view-recovery-development-20261011.json')
RAW = Path('runs/blue-view-recovery-development-20261011.json')


def evaluate():
    if RAW.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve evidence')
    trial = checked_trial()
    paths = {'parent':PARENT,**{k:Path(r['path'])/'student.fp32.tflite' for k,r in trial['trials'].items()}}
    models,identities = {},{}
    for name,path in paths.items():
        meta = read_json(path.with_suffix('.json'))
        if meta['threshold']!=.8 or meta['sha256']!=sha256(path):
            raise ValueError('Model contract changed')
        if name!='parent':
            r = trial['trials'][name]
            if (meta['source_student_sha256']!=r['model_sha256'] or meta['provenance_sha256']!=r['provenance_sha256']
                    or meta['training_trial_sha256']!=sha256(TRIAL) or meta['parity_rows']!=1916
                    or not 0<=meta['max_keras_tflite_delta']<=.002 or meta['parameters']!=7763
                    or not 0<=meta['max_checkpoint_original_group_delta']<=1e-5
                    or meta['export_script_sha256']!=sha256('scripts/export_blue_view_recovery.py')
                    or meta['deployment_approved'] is not False or meta['pi_zero_verified'] is not False):
                raise ValueError('Unverified export')
        models[name] = Predictor(path,allow_unvalidated=True)
        identities[name] = dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
    raw,panels,comparisons = {},{},[]
    for panel in PANELS:
        reference_path = Path(f'runs/real-blue-clutter-scenes-{panel}-20261011.json')
        reference = read_json(reference_path)
        sources,review = reviewed_frames('data/'+panel,'configs/'+panel+'.json')
        if (reference['classifiers']['admit48']!=identities['parent'] or review['threshold']!=.8
                or review['matching_iou']!=.5 or reference['review_sha256']!=sha256('configs/'+panel+'.json')
                or reference['queue_sha256']!=review['queue_sha256']):
            raise ValueError('Frozen parent/panel changed')
        identity = reference['searches']['coarse47']
        path = Path(identity['path'])
        if sha256(path)!=identity['sha256'] or sha256(path.with_suffix('.json'))!=identity['metadata_sha256']:
            raise ValueError('Search model changed')
        grid = GridProposals(path)
        rows = {f'{m}/{s}':[] for m in models for s in ('baseline','coarse47')}
        for i,(r,rgb) in enumerate(sources):
            base = experimental(rgb,'mser')
            proposed = {'baseline':base,'coarse47':supplement(base,grid.predict(rgb))}
            for name,predictor in models.items():
                for search,candidates in proposed.items():
                    if candidates[:len(base)]!=base or len(candidates)>min(12,len(base)+2):
                        raise ValueError('Call budget changed')
                    detections = select_detections(classify_candidates(predictor,rgb,candidates),'split_merged',2.42913421076193)
                    row = dict(id=r['id'],source=r['source'],frame_sha256=r['sha256'],truth=r['provisional_truth'],
                        detections=detections,counts=detection_counts(r['provisional_truth'],detections),extra_calls=len(candidates)-len(base))
                    if name=='parent':
                        check_baseline(row,reference['results'][f'admit48/{search}']['frames'][i])
                    rows[f'{name}/{search}'].append(row)
        raw[panel] = {}
        for name,frames in rows.items():
            counts = {c:{k:sum(f['counts'][c][k] for f in frames) for k in ('tp','fp','fn')} for c in LABELS}
            raw[panel][name] = dict(frames=frames,metrics=metrics(counts))
        values = {k:checked_metrics(v) for k,v in raw[panel].items()}
        for seed in (47,48):
            for search in ('baseline','coarse47'):
                name = f'admit{seed}/{search}'
                losses = []
                for row,old in zip(rows[name],rows[f'parent/{search}'],strict=True):
                    losses.extend(dict(source=row['source'],target=row['truth'][j]) for j in sorted(matched_truth(old)-matched_truth(row)))
                comparisons.append(dict(panel=panel,seed=seed,search=search,
                    nonregression_vs_parent=nonregression(values[name],values[f'parent/{search}']),
                    nonregression_vs_control=nonregression(values[name],values[f'control{seed}/{search}']),
                    lost_parent_targets=losses,meets_target=meets_target(values[name])))
        panels[panel] = dict(frames=len(sources),results=values,reference_sha256=sha256(reference_path),
            queue_sha256=review['queue_sha256'],review_sha256=sha256('configs/'+panel+'.json'),search=identity)
        print(panel,{k:v for k,v in values.items() if k.endswith('coarse47')},flush=True)
    targets = {k:all(meets_target(p['results'][k]) for p in panels.values()) for k in next(iter(raw.values()))}
    write_json(RAW,raw)
    helpers = ('scripts/export_blue_view_recovery.py','scripts/research_balloon_search.py','scripts/research_predictor.py',
        'scripts/pi_balloon_search_bench.py','scripts/evaluate_red_blue_development.py','scripts/audit_balloon_dense_proposals.py',
        'scripts/balloon_grid_runtime.py','scripts/summarize_public_balloon_trial.py')
    write_json(OUTPUT,dict(panels=panels,classifiers=identities,comparisons=comparisons,
        per_variant_all_panels_target=targets,training_trial_sha256=sha256(TRIAL),
        raw=dict(path=str(RAW),sha256=sha256(RAW)),script_sha256=sha256(__file__),helpers_sha256={p:sha256(p) for p in helpers},
        classifier_threshold=.8,matching_iou=.5,selection='split_merged',split_max_aspect=2.42913421076193,
        decision='Independent qualification still required' if any(targets.values()) else 'No fixed variant reaches94%per-color precision and recall on every panel',
        scope='Repeated related development only; no qualification, timing or deployment claim.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(targets=targets))


if __name__=='__main__':
    evaluate()


