"""TRAIN-only fixed two-slot learned-blue reservation, unchanged parent networks."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.evaluate_background_convergence import BASELINE, LOCALIZER, LOCALIZER_SHA, MODEL, MODEL_SHA
from scripts.evaluate_current_confident_containment import checked_reference, nonregressing
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.research_confident_containment import select
from scripts.research_predictor import classify_candidates
from scripts.research_reserved_blue_proposals import reserved
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/reserved-blue-proposals-training-20261011.json')
RAW = Path('runs/reserved-blue-proposals-training-20261011.json')


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    checked_reference()
    reference = read_json(BASELINE)
    if (sha256(reference['raw']['path']) != reference['raw']['sha256']
            or reference['script_sha256'] != sha256('scripts/evaluate_current_confident_containment.py')
            or any(sha256(p) != h for p, h in reference['helpers_sha256'].items())
            or sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA):
        raise ValueError('Frozen baseline changed')
    old = read_json(reference['raw']['path'])['training']
    if checked_metrics(old) != reference['panels']['training']['after']:
        raise ValueError('Baseline metrics changed')
    pixels = {s['source']: (s, rgb) for s, rgb in source_frames() if s['origin'] == 'complete_fit'}
    if len(pixels) != 64 or len(old['frames']) != 64 or any('IMG_' in s for s in pixels):
        raise ValueError('TRAIN boundary changed')
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    model, network = Predictor(MODEL, allow_unvalidated=True), DenseProposals(LOCALIZER)
    if model.metadata['threshold'] != .8:
        raise ValueError('Threshold changed')
    rows = {'parent': [], 'reserved': []}
    changed = []
    for original in old['frames']:
        source, rgb = pixels[original['source']]
        if source['source_file_sha256'] != original['frame_sha256'] or source['truth'] != original['truth']:
            raise ValueError('Source identity changed')
        base = search(rgb, native, True)
        network.predict(rgb)
        learned = decode(network.interpreter.get_tensor(network.output['index'])[0], True)
        policies = dict(parent=supplement(base, learned), reserved=reserved(base, learned))
        if len(policies['reserved']) > len(policies['parent']) or len(policies['reserved']) > 12:
            raise ValueError('Increased classifier call budget')
        if policies['parent'] != policies['reserved']:
            changed.append(original['source'])
        for name, proposals in policies.items():
            detections = select(classify_candidates(model, rgb, proposals))
            row = dict({k: original[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                detections=detections, counts=detection_counts(original['truth'], detections), neural_calls=len(proposals))
            if name == 'parent':
                check_baseline(row, original)
            rows[name].append(row)
    full, summaries = {}, {}
    for name, frames in rows.items():
        counts = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
        full[name] = dict(frames=frames, metrics=metrics(counts))
        checked_metrics(full[name])
        comparison = compare_frames(frames, rows['parent'])
        summaries[name] = dict(metrics=full[name]['metrics'], comparison=comparison,
            nonregression=nonregressing(old['metrics'], full[name]['metrics'], comparison),
            neural_calls=sum(f['neural_calls'] for f in frames))
    write_json(RAW, full)
    write_json(OUTPUT, dict(results=summaries, changed_scenes=changed, raw=dict(path=str(RAW), sha256=sha256(RAW)),
        script_sha256=sha256(__file__), policy_sha256=sha256('scripts/research_reserved_blue_proposals.py'),
        baseline_sha256=sha256(BASELINE), classifier_sha256=MODEL_SHA, localizer_sha256=LOCALIZER_SHA,
        native_sha256=sha256(native.path), eligible_for_development=summaries['reserved']['nonregression'],
        scope='Fixed2slot learned-blue reservation on64completeTRAIN scenes. Parent detections exactly reproduced. Classifier calls never increased; this is not measured Pi speed. No classifier retraining or threshold tuning.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(changed_scenes=len(changed), results={n: dict(metrics=r['metrics'],
        nonregression=r['nonregression'], calls=r['neural_calls'],
        lost=len(r['comparison']['lost_parent_targets']), gained=len(r['comparison']['gained_targets']))
        for n, r in summaries.items()}), flush=True)


if __name__ == '__main__':
    evaluate()
