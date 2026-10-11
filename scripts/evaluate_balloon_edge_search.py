"""Training coverage followed by frozen development evaluation of edge proposals."""
import argparse
from pathlib import Path
import time

from dtr.data import sha256,write_json
from scripts.research_balloon_search import experimental
from scripts.research_balloon_edge_ellipse import proposals
from scripts.audit_balloon_search_stages import original_frames,indoor_frames
from scripts.audit_balloon_pipeline_errors import maximum_matching
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics


BASE='data/balloon-red-blue-bootstrap-20261008/manifest.json'
ORIGINAL='data/raw/matterport-balloon/balloon_dataset.zip'
INDOOR='data/raw/engdes2-red-blue-v1-20261008/dataset.zip'
MODEL='runs/balloon-red-blue-photometric-clutter4-admit-seed47-20261010/student.fp32.tflite'
TRAIN_QUEUES=('engdes2-indoor-review-20261008','engdes2-indoor-expansion-review-20261008',
              'engdes2-indoor-expansion2-review-20261011')


def coverage(truth,candidates):
    return {label:dict(targets=len(selected),localized=len(maximum_matching(
        selected,candidates,lambda t,d:d['color_group']==group)))
        for group,label in enumerate(LABELS)
        for selected in ([r for r in truth if r['label']==label],)}


def base_receipt():
    return dict(search_sha256=sha256('scripts/research_balloon_edge_ellipse.py'),
                baseline_sha256=sha256('scripts/research_balloon_search.py'),
                script_sha256=sha256(__file__),test_evaluated=False,deployment_approved=False,
                pi_timing_measured=False)


def training():
    panels={'original':original_frames(BASE,ORIGINAL)}
    panels.update({name:indoor_frames('data/'+name,'configs/'+name+'.json',BASE,INDOOR)
                   for name in TRAIN_QUEUES})
    results={}
    for name,sources in panels.items():
        frames=[]
        for source,rgb,truth in sources:
            base=experimental(rgb,'mser')
            start=time.perf_counter()
            edge=proposals(rgb)
            elapsed=(time.perf_counter()-start)*1000
            if edge[:len(base)]!=base or len(edge)>min(12,len(base)+2):
                raise ValueError('Baseline or bounded-call contract changed')
            frames.append(dict(source=source,truth=truth,baseline=coverage(truth,base),
                edge=coverage(truth,edge),extra_calls=len(edge)-len(base),host_search_ms=elapsed,
                added_proposals=edge[len(base):]))
        totals={label:dict(targets=sum(f['baseline'][label]['targets'] for f in frames),
            baseline=sum(f['baseline'][label]['localized'] for f in frames),
            edge=sum(f['edge'][label]['localized'] for f in frames)) for label in LABELS}
        results[name]=dict(frames=frames,totals=totals)
    return dict(base_receipt(),panels=results,inputs={p:sha256(p) for p in (BASE,ORIGINAL,INDOOR)},
                training_reviews={name:dict(queue_sha256=sha256('data/'+name+'/review.json'),
                    review_sha256=sha256('configs/'+name+'.json')) for name in TRAIN_QUEUES},
                scope='One-to-one TRAIN proposal coverage ceiling; incomplete scene truth, not detector precision or Pi timing')


def development(queue,review_path):
    from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
    from scripts.research_predictor import load_predictor,classify_candidates
    from scripts.pi_balloon_search_bench import select_detections
    from dtr.vision import RED_BLUE_PROFILE,validate_model_profile
    predictor=load_predictor(MODEL)
    validate_model_profile(predictor.metadata,RED_BLUE_PROFILE)
    frames,review=reviewed_frames(queue,review_path)
    if predictor.metadata['threshold']!=.8 or review['threshold']!=.8 or review['matching_iou']!=.5:
        raise ValueError('Frozen operating point changed')
    outputs={k:[] for k in ('baseline','edge')}
    for row,rgb in frames:
        for name,search in (('baseline',lambda rgb:experimental(rgb,'mser')),('edge',proposals)):
            found=classify_candidates(predictor,rgb,search(rgb))
            detections=select_detections(found,'split_merged',2.42913421076193)
            outputs[name].append(dict(id=row['id'],source=row['source'],frame_sha256=row['sha256'],
                truth=row['provisional_truth'],detections=detections,
                counts=detection_counts(row['provisional_truth'],detections)))
    results={}
    for name,rows in outputs.items():
        counts={label:{key:sum(r['counts'][label][key] for r in rows) for key in ('tp','fp','fn')}
                for label in LABELS}
        results[name]=dict(metrics=metrics(counts),frames=rows)
    return dict(base_receipt(),results=results,model_sha256=sha256(MODEL),
                model_metadata_sha256=sha256(Path(MODEL).with_suffix('.json')),
                queue_sha256=review['queue_sha256'],review_sha256=sha256(review_path),
                threshold=.8,matching_iou=.5,selection='split_merged',split_max_aspect=2.42913421076193,
                scope='Previously inspected development, not independent qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('training','development'),required=True)
    p.add_argument('--queue')
    p.add_argument('--review')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    if a.mode=='development' and not (a.queue and a.review):
        p.error('Development requires reviewed queue')
    report=training() if a.mode=='training' else development(a.queue,a.review)
    write_json(a.output,report)
    print({k:v['totals'] for k,v in report['panels'].items()} if a.mode=='training'
          else {k:v['metrics'] for k,v in report['results'].items()})
