"""Fixed dense-supplement comparison on reviewed development scenes, not qualification."""
import argparse
from pathlib import Path
import time

from dtr.data import sha256,write_json
from dtr.runtime import Predictor
from scripts.balloon_dense_runtime import DenseProposals
from scripts.audit_balloon_dense_proposals import supplement
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_red_blue_development import LABELS,detection_counts,metrics

CLASSIFIER='runs/balloon-fragment-control-seed47-20261011/color.fp32.tflite'


def evaluate(queue,review_path,model_paths=None):
    frames,review=reviewed_frames(queue,review_path)
    if review['threshold']!=.8 or review['matching_iou']!=.5:
        raise ValueError('Review operating point changed')
    predictor=Predictor(CLASSIFIER,allow_unvalidated=True)
    if predictor.metadata['threshold']!=.8:
        raise ValueError('Classifier threshold changed')
    paths={str(seed):Path(f'runs/balloon-dense-proposals-seed{seed}-20261011/proposal.fp32.tflite') for seed in (47,48)}
    if model_paths is not None:
        if set(model_paths)!={'47','48'}:
            raise ValueError('Require both fixed-seed proposal models')
        paths={seed:Path(path) for seed,path in model_paths.items()}
    networks={seed:DenseProposals(path) for seed,path in paths.items()}
    output={name:[] for name in ('baseline','dense47','dense48')}
    for row,rgb in frames:
        base=experimental(rgb,'mser')
        for name in output:
            started=time.perf_counter()
            proposed=base if name=='baseline' else supplement(base,networks[name[-2:]].predict(rgb))
            dense_ms=(time.perf_counter()-started)*1000 if name!='baseline' else 0.
            if proposed[:len(base)]!=base or len(proposed)>min(12,len(base)+2):
                raise ValueError('Baseline or proposal budget changed')
            found=classify_candidates(predictor,rgb,proposed)
            detections=select_detections(found,'split_merged',2.42913421076193)
            output[name].append(dict(id=row['id'],source=row['source'],frame_sha256=row['sha256'],
                truth=row['provisional_truth'],detections=detections,extra_calls=len(proposed)-len(base),
                dense_host_ms=dense_ms,counts=detection_counts(row['provisional_truth'],detections)))
    results={}
    for name,rows in output.items():
        counts={label:{key:sum(r['counts'][label][key] for r in rows) for key in ('tp','fp','fn')} for label in LABELS}
        results[name]=dict(frames=rows,metrics=metrics(counts))
    return dict(results=results,classifier_sha256=sha256(CLASSIFIER),
        dense_models={seed:dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
                      for seed,path in paths.items()},
        queue_sha256=review['queue_sha256'],review_sha256=sha256(review_path),
        helpers_sha256={p:sha256(p) for p in ('scripts/balloon_dense_runtime.py','scripts/audit_balloon_dense_proposals.py',
            'scripts/research_balloon_search.py','scripts/pi_balloon_search_bench.py','scripts/evaluate_red_blue_development.py')},
        script_sha256=sha256(__file__),selection='split_merged',split_max_aspect=2.42913421076193,
        proposal_threshold=.1,classifier_threshold=.8,matching_iou=.5,
        scope='Repeatedly inspected development, not independent qualification. Same classifier, at most two added blue proposals, twelve crops total.',
        test_evaluated=False,deployment_approved=False,pi_tested=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('queue','review','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--dense47')
    p.add_argument('--dense48')
    a=p.parse_args()
    if bool(a.dense47)!=bool(a.dense48):
        p.error('Provide both --dense47 and --dense48, or neither')
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    paths={'47':a.dense47,'48':a.dense48} if a.dense47 else None
    report=evaluate(a.queue,a.review,paths)
    write_json(a.output,report)
    print({k:v['metrics'] for k,v in report['results'].items()})
