"""Frozen scene comparison of matched fragment students on MSER and Lab search."""
import argparse
from pathlib import Path

from dtr.data import sha256,write_json
from dtr.vision import RED_BLUE_PROFILE,validate_model_profile
from scripts.balloon_fragment_runtime import FragmentPredictor
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics
from scripts.pi_balloon_search_bench import select_detections
from scripts.research_balloon_search import experimental
from scripts.research_balloon_color_split import proposals
from scripts.research_predictor import load_predictor,classify_candidates
from scripts.train_balloon_fragments import PARENT


def evaluate(queue,review_path):
    frames,review=reviewed_frames(queue,review_path)
    if review['threshold']!=.8 or review['matching_iou']!=.5:
        raise ValueError('Frozen operating point changed')
    sources={'incumbent':str(Path(PARENT)/'student.fp32.tflite')}
    sources.update({f'{mode}{seed}':f'runs/balloon-fragment-{mode}-seed{seed}-20261011/fragment.fp32.tflite'
                    for mode in ('control','fragment') for seed in (47,48)})
    candidates={row['id']:dict(mser=experimental(rgb,'mser'),lab=proposals(rgb)) for row,rgb in frames}
    results={}
    for name,path in sources.items():
        predictor=load_predictor(path) if name=='incumbent' else FragmentPredictor(path)
        if name=='incumbent':
            validate_model_profile(predictor.metadata,RED_BLUE_PROFILE)
        if predictor.metadata['threshold']!=.8:
            raise ValueError('Classifier threshold changed')
        variants={}
        for search in ('mser','lab'):
            outputs=[]
            for row,rgb in frames:
                found=classify_candidates(predictor,rgb,candidates[row['id']][search])
                detections=select_detections(found,'split_merged',2.42913421076193)
                outputs.append(dict(id=row['id'],source=row['source'],frame_sha256=row['sha256'],
                    truth=row['provisional_truth'],detections=detections,
                    counts=detection_counts(row['provisional_truth'],detections)))
            totals={label:{k:sum(r['counts'][label][k] for r in outputs) for k in ('tp','fp','fn')}
                    for label in LABELS}
            variants[search]=dict(metrics=metrics(totals),frames=outputs,
                fragment_argmax_rejections=sum(d.get('fragment_rejected',False) for r in outputs for d in r['detections']))
        results[name]=dict(model_sha256=sha256(path),metadata_sha256=sha256(Path(path).with_suffix('.json')),
                           variants=variants)
    return dict(results=results,queue_sha256=review['queue_sha256'],review_sha256=sha256(review_path),
        script_sha256=sha256(__file__),predictor_sha256=sha256('scripts/balloon_fragment_runtime.py'),
        search_sha256={p:sha256(p) for p in ('scripts/research_balloon_search.py','scripts/research_balloon_color_split.py')},
        selection='split_merged',split_max_aspect=2.42913421076193,threshold=.8,matching_iou=.5,
        scope='Repeatedly inspected development scenes; no independent qualification. Fixed final epoch and unchanged operating point.',
        test_evaluated=False,deployment_approved=False,pi_timing_measured=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('queue','review','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    report=evaluate(a.queue,a.review)
    write_json(a.output,report)
    print({k:{search:v['metrics'] for search,v in r['variants'].items()} for k,r in report['results'].items()})
