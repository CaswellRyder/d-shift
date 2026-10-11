"""Fixed TRAIN screen for boundary-censored free-center ellipse proposals."""
from pathlib import Path
import time

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.compare_current_blue_search import MODEL, MODEL_SHA, RAW as PREVIOUS, OUTPUT as PREVIOUS_RECEIPT, gate
from scripts.audit_current_blue_search import matched_truth
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_balloon_search import experimental
from scripts.research_censored_ellipse import proposals
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.summarize_public_balloon_trial import checked_metrics

RAW = Path('runs/censored-ellipse-training-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/censored-ellipse-training-20261011.json')


def evaluate():
    if RAW.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve previous evidence')
    previous = read_json(PREVIOUS_RECEIPT)
    if (sha256(MODEL) != MODEL_SHA or sha256(PREVIOUS) != previous['raw']['sha256']
            or any(sha256(p) != h for p,h in previous['helpers_sha256'].items())):
        raise ValueError('Frozen inputs changed')
    old = read_json(PREVIOUS)['results']['mser']
    refs = {f['source']: f for f in old['frames']}
    predictor = Predictor(MODEL, allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Operating point changed')
    frames, losses, gains = [], [], []
    for source, rgb in source_frames():
        if source['origin'] != 'complete_fit':
            continue
        ref = refs[source['source']]
        if ref['truth'] != source['truth'] or ref['frame_sha256'] != source['source_file_sha256']:
            raise ValueError('Source identity changed')
        base = experimental(rgb,'mser')
        start = time.perf_counter()
        candidate = proposals(rgb)
        elapsed = (time.perf_counter()-start)*1000
        if candidate[:len(base)] != base or len(candidate) > min(12,len(base)+2):
            raise ValueError('Call cap/baseline violated')
        found = classify_candidates(predictor,rgb,candidate)
        detections = select_detections(found,'split_merged',2.42913421076193)
        # Check same baseline inference before any joint postprocessing.
        baseline_detections = select_detections(found[:len(base)],'split_merged',2.42913421076193)
        stripped = lambda rows: [{k:v for k,v in r.items() if k != 'latency_ms'} for r in rows]
        if stripped(baseline_detections) != stripped(ref['detections']):
            raise ValueError('Baseline predictions changed')
        row = dict(source=source['source'],frame_sha256=source['source_file_sha256'],truth=source['truth'],
            detections=detections,counts=detection_counts(source['truth'],detections),
            calls=len(candidate),extra_calls=len(candidate)-len(base),host_search_ms=elapsed)
        before, after = matched_truth(ref), matched_truth(row)
        losses.extend(dict(source=row['source'],target=row['truth'][i]) for i in sorted(before-after))
        gains.extend(dict(source=row['source'],target=row['truth'][i]) for i in sorted(after-before))
        frames.append(row)
    if len(frames) != 64 or {r['source'] for r in frames} != set(refs):
        raise ValueError('Incomplete TRAIN screen')
    counts = {c:{k:sum(f['counts'][c][k] for f in frames) for k in ('tp','fp','fn')} for c in LABELS}
    result = dict(frames=frames,metrics=metrics(counts))
    checked = checked_metrics(result)
    passed = gate(checked_metrics(old),checked) and not losses
    write_json(RAW,result)
    write_json(OUTPUT,dict(metrics=checked,baseline=checked_metrics(old),training_gate=passed,
        gained_targets=gains,lost_targets=losses,calls=sum(r['calls'] for r in frames),
        extra_calls=sum(r['extra_calls'] for r in frames),max_extra_calls=max(r['extra_calls'] for r in frames),
        host_mean_search_ms=sum(r['host_search_ms'] for r in frames)/len(frames),
        raw=dict(path=str(RAW),sha256=sha256(RAW)),previous_receipt_sha256=sha256(PREVIOUS_RECEIPT),
        classifier_sha256=MODEL_SHA,classifier_metadata_sha256=sha256(MODEL.with_suffix('.json')),
        scripts_sha256={p:sha256(p) for p in (__file__,'scripts/research_censored_ellipse.py',
            'scripts/compare_current_blue_search.py','scripts/audit_current_blue_search.py')},
        helpers_sha256=previous['helpers_sha256'],frames=64,threshold=.8,matching_iou=.5,
        scope='Fixed TRAIN geometry experiment. Host timing includes search only, not Pi or camera. No independent accuracy claim.',
        decision='Eligible for separate development evaluation' if passed else 'Rejected at TRAIN gate; no runtime change',
        development_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(metrics=checked,training_gate=passed,gained=len(gains),lost=len(losses),
        calls=sum(r['calls'] for r in frames)))


if __name__ == '__main__':
    evaluate()
