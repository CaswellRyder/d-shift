"""Recheck existing bounded blue search alternatives with the improved classifier.

Use only64 previously complete-reviewed FIT scenes. This is a TRAIN screen,
not a development threshold sweep, Pi timing test or qualification.
"""
from pathlib import Path

from dtr.data import sha256, write_json
from dtr.runtime import Predictor
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.summarize_public_balloon_trial import checked_metrics

MODEL = Path('runs/real-blue-clutter-convergence-admit-seed48-20261011/student.fp32.tflite')
MODEL_SHA = '62ecb93da3d3e6f94ca0e1213520fa04191d0ad5666810e489ba0ed26d4ff3c1'
MODES = ('mser', 'mser_blue_cores', 'mser_blue_ellipse')
OUTPUT = Path('research/balloon-red-blue-20261008/current-blue-search-training-20261011.json')
RAW = Path('runs/current-blue-search-training-20261011.json')


def gate(baseline, candidate):
    if set(baseline) != set(LABELS) or set(candidate) != set(LABELS):
        raise ValueError('Both colors required')
    for color in LABELS:
        if baseline[color]['tp'] + baseline[color]['fn'] != candidate[color]['tp'] + candidate[color]['fn']:
            raise ValueError('Target count changed')
    return (all(candidate[c]['tp'] >= baseline[c]['tp'] and candidate[c]['fp'] <= baseline[c]['fp'] for c in LABELS)
            and candidate['blue_balloon']['tp'] > baseline['blue_balloon']['tp'])


def compare():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Do not overwrite search comparison')
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Frozen classifier changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Operating point changed')
    results = {mode: [] for mode in MODES}
    inputs = []
    for source, rgb in source_frames():
        if source['origin'] != 'complete_fit':
            continue
        inputs.append(source)
        base = experimental(rgb, 'mser')
        for mode in MODES:
            proposals = base if mode == 'mser' else experimental(rgb, mode)
            if proposals[:len(base)] != base or len(proposals) > min(12, len(base) + 2):
                raise ValueError('Bounded baseline-preserving search contract changed')
            found = classify_candidates(predictor, rgb, proposals)
            detected = select_detections(found, 'split_merged', 2.42913421076193)
            results[mode].append(dict(source=source['source'], frame_sha256=source['source_file_sha256'],
                truth=source['truth'], detections=detected, calls=len(proposals), extra_calls=len(proposals) - len(base),
                counts=detection_counts(source['truth'], detected)))
    if len(inputs) != 64:
        raise ValueError('Require exactly64 complete FIT scenes')
    summary, raw = {}, {}
    for mode, frames in results.items():
        counts = {c: {key: sum(f['counts'][c][key] for f in frames) for key in ('tp', 'fp', 'fn')} for c in LABELS}
        raw[mode] = dict(frames=frames, metrics=metrics(counts))
        summary[mode] = dict(metrics=checked_metrics(raw[mode]), calls=sum(f['calls'] for f in frames),
            extra_calls=sum(f['extra_calls'] for f in frames), max_extra_calls=max(f['extra_calls'] for f in frames))
    passed = {mode: gate(summary['mser']['metrics'], summary[mode]['metrics']) for mode in MODES[1:]}
    write_json(RAW, dict(results=raw, source_frames=inputs))
    write_json(OUTPUT, dict(results=summary, training_gate=passed, raw=dict(path=str(RAW), sha256=sha256(RAW)),
        classifier=dict(path=str(MODEL), sha256=MODEL_SHA, metadata_sha256=sha256(MODEL.with_suffix('.json'))),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/prepare_blue_false_alarm_review.py', 'scripts/research_balloon_search.py',
            'scripts/research_predictor.py', 'scripts/pi_balloon_search_bench.py',
            'scripts/evaluate_red_blue_development.py', 'scripts/summarize_public_balloon_trial.py')},
        frames=64, classifier_threshold=.8, matching_iou=.5, max_total_proposals=12, max_extra_proposals=2,
        selection='split_merged', split_max_aspect=2.42913421076193,
        gate_description='Strict blue TP gain, no red TP loss, no FP increase for either color, same64 complete FIT scenes.',
        scope='Existing search alternatives, fixed improved classifier and TRAIN operating point. No model retraining, independent accuracy, development tuning or Pi timing claim.',
        test_evaluated=False, development_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(results=summary, training_gate=passed))


if __name__ == '__main__':
    compare()
