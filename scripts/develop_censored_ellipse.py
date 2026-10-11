"""Evaluate a frozen TRAIN-gated search on existing development panels."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_censored_ellipse import OUTPUT as TRAIN, RAW as TRAIN_RAW
from scripts.research_censored_ellipse import proposals
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.audit_current_blue_search import matched_truth

PANELS = ('engdes2-lockbox-panel3-20261010','engdes2-development-panel2-20261010','engdes2-development-scenes-20261008')
RAW = Path('runs/censored-ellipse-development-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/censored-ellipse-development-20261011.json')


def evaluate():
    if RAW.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve evidence')
    train = read_json(TRAIN)
    if (train['training_gate'] is not True or sha256(TRAIN_RAW) != train['raw']['sha256']
            or sha256(MODEL) != MODEL_SHA or sha256(MODEL.with_suffix('.json')) != train['classifier_metadata_sha256']
            or any(sha256(p) != h for p,h in {**train['scripts_sha256'],**train['helpers_sha256']}.items())):
        raise ValueError('Frozen TRAIN gate/provenance changed')
    predictor = Predictor(MODEL,allow_unvalidated=True)
    results, summary = {}, {}
    for panel in PANELS:
        sources, review = reviewed_frames('data/'+panel,'configs/'+panel+'.json')
        if review['threshold'] != .8 or review['matching_iou'] != .5:
            raise ValueError('Operating point changed')
        rows = {'baseline':[],'candidate':[]}
        losses, gains = [], []
        for source,rgb in sources:
            baseline = experimental(rgb,'mser')
            candidate = proposals(rgb)
            if candidate[:len(baseline)] != baseline or len(candidate) > min(12,len(baseline)+2):
                raise ValueError('Bounded baseline contract changed')
            found = classify_candidates(predictor,rgb,candidate)
            for name,count in (('baseline',len(baseline)),('candidate',len(candidate))):
                detected = select_detections(found[:count],'split_merged',2.42913421076193)
                rows[name].append(dict(id=source['id'],source=source['source'],frame_sha256=source['sha256'],
                    truth=source['provisional_truth'],detections=detected,calls=count,
                    counts=detection_counts(source['provisional_truth'],detected)))
            old,new = matched_truth(rows['baseline'][-1]),matched_truth(rows['candidate'][-1])
            for dest,indices in ((losses,old-new),(gains,new-old)):
                dest.extend(dict(source=source['source'],target=source['provisional_truth'][i]) for i in sorted(indices))
        results[panel] = {}
        for name,frames in rows.items():
            counts = {c:{k:sum(f['counts'][c][k] for f in frames) for k in ('tp','fp','fn')} for c in LABELS}
            results[panel][name] = dict(frames=frames,metrics=metrics(counts))
        values = {n:checked_metrics(r) for n,r in results[panel].items()}
        passed = (not losses and all(values['candidate'][c]['tp'] >= values['baseline'][c]['tp']
                    and values['candidate'][c]['fp'] <= values['baseline'][c]['fp'] for c in LABELS))
        summary[panel] = dict(results=values,frames=len(sources),non_regression=passed,
            gained_targets=gains,lost_targets=losses,
            extra_calls=sum(f['calls'] for f in rows['candidate'])-sum(f['calls'] for f in rows['baseline']),
            queue_sha256=review['queue_sha256'],review_sha256=sha256('configs/'+panel+'.json'))
        print(panel,summary[panel],flush=True)
    write_json(RAW,results)
    passed = all(p['non_regression'] for p in summary.values()) and any(p['gained_targets'] for p in summary.values())
    write_json(OUTPUT,dict(panels=summary,development_gate=passed,training_receipt_sha256=sha256(TRAIN),
        raw=dict(path=str(RAW),sha256=sha256(RAW)),script_sha256=sha256(__file__),
        decision='Eligible for separate isolated Pi speed check' if passed else 'Rejected on development; no deployment',
        scope='Repeated overlapping development panels, not independent held-out qualification. No thresholds selected.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print('development_gate',passed)


if __name__ == '__main__':
    evaluate()
