"""Compare fixed learned selection on training and frozen development separately."""
import argparse
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from dtr.runtime import Predictor
from scripts.balloon_dense_runtime import DenseProposals
from scripts.research_dense_supported_selection import supported_selection,candidate_pairs
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.evaluate_red_blue_development import detection_counts,metrics,LABELS
from scripts.balloon_expansion3 import checked_extra,QUEUE,REVIEW,BOOTSTRAP
from scripts.audit_balloon_search_stages import indoor_frames
from scripts.audit_balloon_expansion3_failures import explain
from scripts.train_balloon_reviewed_clutter import PARENT


def evaluate(training,queue=None,review=None):
    if training:
        checked_extra()
        source=indoor_frames(QUEUE,REVIEW,BOOTSTRAP,'data/raw/engdes2-red-blue-v1-20261008/dataset.zip')
        frames=[(dict(id=i,source=name,provisional_truth=truth),rgb)
                for i,(name,rgb,truth) in enumerate(source)]
        queue,review=QUEUE,REVIEW
    else:
        frames,decision=reviewed_frames(queue,review)
        if decision['threshold']!=.8 or decision['matching_iou']!=.5:
            raise ValueError('Operating point changed')
    classifier=PARENT/'color.fp32.tflite'
    metadata=read_json(classifier.with_suffix('.json'))
    if metadata['threshold']!=.8 or metadata['sha256']!=sha256(classifier):
        raise ValueError('Classifier identity or threshold changed')
    predictor=Predictor(classifier,allow_unvalidated=True)
    paths={str(s):Path(f'runs/balloon-dense-overlap-l1_giou-seed{s}-20261011/proposal.fp32.tflite') for s in (47,48)}
    networks={s:DenseProposals(p) for s,p in paths.items()}
    results={n:[] for n in ('baseline','supported47','supported48')}
    for row,rgb in frames:
        raw=classify_candidates(predictor,rgb,experimental(rgb,'mser'))
        variants=dict(baseline=select_detections(raw,'split_merged',2.42913421076193))
        needs_dense=bool(candidate_pairs(raw))
        for seed,net in networks.items():
            variants['supported'+seed]=supported_selection(raw,net.predict(rgb) if needs_dense else [])
        for name,found in variants.items():
            if len(found)!=len(raw) or any(a['box']!=b['box'] or a['scores']!=b['scores']
                                          for a,b in zip(found,raw,strict=True)):
                raise ValueError('Selection changed proposals or scores')
            item=dict(id=row['id'],source=row['source'],truth=row['provisional_truth'],detections=found,
                dense_calls=int(name!='baseline' and needs_dense),
                changed_acceptances=sum(a['accepted']!=b['accepted'] for a,b in zip(found,variants['baseline'],strict=True)))
            if training:
                item['stages']=[dict(label=t['label'],stage=explain(t,found)) for t in item['truth']]
            else:
                item['frame_sha256']=row['sha256']
                item['counts']=detection_counts(item['truth'],found)
            results[name].append(item)
    output={}
    for name,rows in results.items():
        entry=dict(frames=rows,changed_acceptances=sum(r['changed_acceptances'] for r in rows),
            dense_calls=sum(r['dense_calls'] for r in rows))
        if training:
            from collections import Counter
            entry['stage_counts']={label:dict(Counter(t['stage'] for r in rows for t in r['stages'] if t['label']==label)) for label in LABELS}
        else:
            entry['metrics']=metrics({label:{k:sum(r['counts'][label][k] for r in rows) for k in ('tp','fp','fn')} for label in LABELS})
        output[name]=entry
    return dict(results=output,classifier_sha256=sha256(classifier),
        dense_models={s:dict(path=str(p),sha256=sha256(p)) for s,p in paths.items()},
        queue_sha256=sha256(Path(queue)/'review.json'),review_sha256=sha256(review),
        script_sha256=sha256(__file__),policy_sha256=sha256('scripts/research_dense_supported_selection.py'),
        training_only=training,scope='Training diagnostic: incomplete truth' if training else 'Repeated development, not qualification',
        extra_crop_calls=0,max_extra_dense_calls_per_frame=1,
        dense_call_gate='Raw-accepted same-color nested region pair; exact necessary condition, not a learned rejection threshold.',
        test_evaluated=False,deployment_approved=False,pi_tested=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',action='store_true')
    p.add_argument('--queue')
    p.add_argument('--review')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or (not a.training and (not a.queue or not a.review)):
        p.error('Require new output and training or queue/review')
    result=evaluate(a.training,a.queue,a.review)
    write_json(a.output,result)
    print({n:{k:v for k,v in r.items() if k!='frames'} for n,r in result['results'].items()})
