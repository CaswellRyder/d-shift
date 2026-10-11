"""TRAIN-only complete-scene error diagnosis. Failed models remain rejected."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor, softmax
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.window2_scene_labels import REVIEW, checked_frames
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_scene_decision_retention import LOCALIZER, LOCALIZER_SHA
from scripts.balloon_dense_runtime import DenseProposals
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_predictor import classify_candidates
from scripts.research_confident_containment import select
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_pipeline_errors import analyze_frame

OUTPUT = Path('research/balloon-red-blue-20261008/window2-complete-scenes-20261011.json')
RAW = Path('runs/window2-complete-scenes-20261011.json')
PRIOR = Path('research/balloon-red-blue-20261008/window2-retraining-20261011.json')


def diagnose(result):
    counts = dict(false_negatives=Counter(), false_positives=Counter())
    frames = []
    for frame in result['frames']:
        detail = analyze_frame(frame)
        for kind in counts:
            for row in detail[kind]:
                counts[kind][row['reason']] += 1
        frames.append(dict(id=frame['id'], detail=detail))
    return dict(counts={k: dict(v) for k, v in counts.items()}, frames=frames)


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve scene diagnosis')
    reviewed = checked_frames()
    if sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Frozen models changed')
    classifier, localizer = Predictor(MODEL, allow_unvalidated=True), DenseProposals(LOCALIZER)
    if classifier.metadata['threshold'] != .8:
        raise ValueError('Classifier threshold changed')
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    frames, pixels = [], []
    for frame, rgb in reviewed:
        proposed = search(rgb, native, True)
        localizer.predict(rgb)
        learned = decode(localizer.interpreter.get_tensor(localizer.output['index'])[0], True)
        proposed = supplement(proposed, learned)
        found = select(classify_candidates(classifier, rgb, proposed))
        frames.append(dict(frame, detections=found, counts=detection_counts(frame['truth'], found),
                           neural_calls=len(proposed)))
        for row in found:
            a, b, c, d = row['crop_box']
            pixels.append(np.asarray(Image.fromarray(rgb[b:d, a:c]).resize((64, 64), Image.Resampling.BILINEAR), np.float32))
    parent = dict(frames=frames, metrics=metrics({c: {k: sum(f['counts'][c][k] for f in frames)
        for k in ('tp', 'fp', 'fn')} for c in LABELS}))
    checked_metrics(parent)
    x = np.stack(pixels)
    import keras
    reference = Path('runs/real-blue-clutter-convergence-admit-seed48-20261011/student.keras')
    if sha256(reference) != '6187512f17e069ae4665134d404ea863062d2561b32e7f5dcfbbb69bceb1c5d8':
        raise ValueError('Parent Keras changed')
    reproduced = replay(frames, softmax(keras.models.load_model(reference, compile=False)(x, training=False).numpy()))
    for a, b in zip(reproduced['frames'], frames, strict=True):
        check_replayed_parent(a, b)
    results = dict(parent=parent)
    models = {str(reference): sha256(reference)}
    prior = read_json(PRIOR)
    for arm in ('control', 'admit'):
        for seed in (47, 48):
            name = f'{arm}{seed}'
            p = Path(prior['results'][name]['run']) / 'student.keras'
            expected = prior['results'][name]['artifacts_sha256']['student.keras']
            if sha256(p) != expected:
                raise ValueError('Audited model changed')
            models[str(p)] = expected
            model = keras.models.load_model(p, compile=False)
            results[name] = replay(frames, softmax(model(x, training=False).numpy()))
    summaries = {}
    for name, result in results.items():
        summaries[name] = dict(metrics=checked_metrics(result), diagnosis=diagnose(result),
                               effect=difference(result, parent))
        print(dict(name=name, metrics=result['metrics'], errors=summaries[name]['diagnosis']['counts']), flush=True)
    write_json(RAW, results)
    inputs = [REVIEW, PRIOR, MODEL, LOCALIZER, __file__, 'scripts/window2_scene_labels.py',
        'scripts/research_blue_bright_search.py', 'scripts/research_blue_peak_blend.py',
        'scripts/research_confident_containment.py', 'scripts/research_predictor.py',
        'scripts/audit_balloon_dense_proposals.py', 'scripts/diagnose_clutter_round3_feature.py',
        'scripts/audit_balloon_pipeline_errors.py', 'scripts/summarize_public_balloon_trial.py', native.path]
    write_json(OUTPUT, dict(results=summaries, frames=len(frames), targets=sum(len(f['truth']) for f in frames),
        proposal_count=len(pixels), raw=dict(path=str(RAW), sha256=sha256(RAW)),
        inputs_sha256={str(p): sha256(p) for p in inputs}, models_sha256=models,
        scope='12 previously admitted public TRAIN photos,29 manually reviewed red/blue targets. Parent full-frame search plus fixed-proposal classifier replay for four already-rejected retraining arms. Diagnostic only: does NOT override failed previous gates. No new export, DEV, reserved-test or hardware execution. Assistant labels require independent review for qualification.',
        decision='Retain failed model status; use TRAIN error attribution to choose the next intervention.',
        training_performed=False, development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
