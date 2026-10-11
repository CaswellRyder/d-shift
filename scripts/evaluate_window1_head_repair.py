"""Compare audited head repair through fresh TRAIN search, then repeated development."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_current_confident_containment import OUTPUT as BASELINE, checked_reference, nonregressing
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.export_window1_head_repair import TRIAL, checked_trial
from scripts.window1_runtime_data import checked_batch
from scripts.audit_window1_scene_search import REVIEW as SCENE_REVIEW
from scripts.window1_trial_data import recovered_targets
from scripts.audit_window1_scene_search import OUTPUT as BATCH_SCENES
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.research_confident_containment import select
from scripts.research_predictor import classify_candidates
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/window1-head-repair-development-20261011.json')
RAW = Path('runs/window1-head-repair-development-20261011.json')
LOCALIZER = Path('runs/blue-head-only-background-20261011/proposal.fp32.tflite')
LOCALIZER_SHA = '1b8123826bae8e81cd933f8c5d7705bb540e0be1eb3135d67c695298450f79a3'


def batch_inputs():
    _, _, frames, _ = checked_batch()
    review = read_json(SCENE_REVIEW)
    root = Path(review['queue'])
    queue = read_json(root/'review.json')
    result = {}
    for frame in frames:
        row = queue['frames'][frame['id']]
        with Image.open(checked_path(root, row['path'], row['sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        result[frame['source']] = (row['sha256'], frame['truth'], rgb)
    return result


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    trial, names = checked_trial()
    checked_reference()
    reference = read_json(BASELINE)
    if (sha256(reference['raw']['path']) != reference['raw']['sha256']
            or reference['script_sha256'] != sha256('scripts/evaluate_current_confident_containment.py')
            or any(sha256(p) != h for p, h in reference['helpers_sha256'].items())
            or sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA):
        raise ValueError('Frozen pipeline changed')
    previous = read_json(reference['raw']['path'])
    if len(previous) != 4 or 'training' not in previous:
        raise ValueError('Unexpected evaluation panels')
    batch_reference = read_json(BATCH_SCENES)
    if sha256(batch_reference['raw']['path']) != batch_reference['raw']['sha256']:
        raise ValueError('Batch TRAIN baseline changed')
    previous['window1-training'] = read_json(batch_reference['raw']['path'])
    reference['panels']['window1-training'] = dict(after=batch_reference['metrics'])
    models = {'parent': Predictor(MODEL, allow_unvalidated=True)}
    exports = {}
    for name in names:
        run = Path(trial['results'][name]['run'])
        path = run/'student.fp32.tflite'
        meta = read_json(path.with_suffix('.json'))
        with np.load(run/'constraints.npz', allow_pickle=False) as archive:
            expected_parity_rows = len(archive['x'])+11
        if (sha256(path) != meta['sha256'] or meta['parameters'] != 7763
                or meta['source_student_sha256'] != trial['results'][name]['model_sha256']
                or meta['training_trial_sha256'] != sha256(TRIAL)
                or meta['export_script_sha256'] != sha256('scripts/export_window1_head_repair.py')
                or meta['parity_rows'] != expected_parity_rows or meta['decision_parity'] is not True
                or not 0 <= meta['max_keras_tflite_delta'] <= .002
                or meta['deployment_approved'] is not False):
            raise ValueError('Classifier export changed')
        models[name] = Predictor(path, allow_unvalidated=True)
        exports[name] = dict(path=str(path), metadata_sha256=sha256(path.with_suffix('.json')), **meta)
    if any(m.metadata['threshold'] != .8 for m in models.values()):
        raise ValueError('Fixed threshold changed')
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    localizer = DenseProposals(LOCALIZER)
    train = {s['source']: (s['source_file_sha256'], s['truth'], rgb)
             for s, rgb in source_frames() if s['origin'] == 'complete_fit'}
    if len(train) != 64 or any('IMG_' in s for s in train):
        raise ValueError('TRAIN boundary changed')
    batch_pixels = batch_inputs()
    inputs, full, panels = {str(SCENE_REVIEW):sha256(SCENE_REVIEW)}, {}, {}
    survivors = names
    for panel in ['training', 'window1-training']+[p for p in previous if p not in ('training', 'window1-training')]:
        if panel != 'training' and not survivors:
            break
        old = previous[panel]
        if checked_metrics(old) != reference['panels'][panel]['after']:
            raise ValueError('Baseline counts changed')
        pixels = batch_pixels if panel == 'window1-training' else train
        if panel not in ('training', 'window1-training'):
            review_path = 'configs/'+panel+'.json'
            sources, review = reviewed_frames('data/'+panel, review_path)
            if review['threshold'] != .8 or review['matching_iou'] != .5:
                raise ValueError('Evaluation operating point changed')
            pixels = {s['source']: (s['sha256'], s['provisional_truth'], rgb) for s, rgb in sources}
            inputs[review_path] = sha256(review_path)
        if set(pixels) != {f['source'] for f in old['frames']}:
            raise ValueError('Scene coverage changed')
        rows = {n: [] for n in ['parent']+survivors}
        for frame in old['frames']:
            digest, truth, rgb = pixels[frame['source']]
            if digest != frame['frame_sha256'] or truth != frame['truth']:
                raise ValueError('Scene identity changed')
            base = search(rgb, native, True)
            localizer.predict(rgb)
            learned = decode(localizer.interpreter.get_tensor(localizer.output['index'])[0], True)
            proposed = supplement(base, learned)
            if proposed[:len(base)] != base or len(proposed) > min(12, len(base)+2):
                raise ValueError('Proposal budget changed')
            for name in rows:
                detections = select(classify_candidates(models[name], rgb, proposed))
                row = dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                    detections=detections, counts=detection_counts(truth, detections), neural_calls=len(proposed))
                if name == 'parent':
                    check_baseline(row, frame)
                rows[name].append(row)
        full[panel], panels[panel] = {}, {}
        for name, frames in rows.items():
            total = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
            result = dict(frames=frames, metrics=metrics(total))
            checked_metrics(result)
            comparison = compare_frames(frames, rows['parent'])
            full[panel][name] = result
            panels[panel][name] = dict(metrics=result['metrics'], comparison=comparison,
                nonregression=nonregressing(old['metrics'], result['metrics'], comparison),
                meets_94=all(result['metrics'][c][m] is not None and result['metrics'][c][m] >= .94 for c in LABELS for m in ('precision', 'recall')),
                neural_calls=sum(f['neural_calls'] for f in frames))
        print(dict(panel=panel, results={n: dict(metrics=r['metrics'], nonregression=r['nonregression'],
            lost=len(r['comparison']['lost_parent_targets']), gained=len(r['comparison']['gained_targets']))
            for n, r in panels[panel].items()}), flush=True)
        if panel == 'training':
            survivors = names if all(panels[panel][n]['nonregression'] for n in names) else []
        if panel == 'window1-training':
            if not all(panels[panel][n]['nonregression'] and recovered_targets(full[panel][n]) for n in names):
                survivors = []
    qualified = [n for n in survivors if len(panels) == 5 and all(p[n]['nonregression'] for p in panels.values())]
    write_json(RAW, full)
    write_json(OUTPUT, dict(panels=panels, eligible_for_development=survivors,
        eligible_for_pi_comparison=qualified,
        exports=exports, localizer_sha256=LOCALIZER_SHA, parent_sha256=MODEL_SHA,
        native_sha256=sha256(native.path), inputs=inputs,
        baseline_sha256=sha256(BASELINE), training_audit_sha256=sha256(TRIAL), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in (
            'scripts/research_blue_bright_search.py', 'scripts/research_blue_peak_blend.py',
            'scripts/research_confident_containment.py', 'scripts/research_predictor.py',
            'scripts/audit_balloon_dense_proposals.py', 'scripts/audit_indoor5_outcomes.py')},
        raw=dict(path=str(RAW), sha256=sha256(RAW)),
        scope='Two independently trained/repaired seeds versus original parent on64old and6newTRAIN scenes, then three related repeated development panels only after both seeds pass fresh TRAIN recovery/nonregression. Same proposals,0.8threshold,IoU0.5andcontainment. No independent final holdout or timing claim.',
        development_evaluated=any(p not in ('training', 'window1-training') for p in panels), test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    evaluate()




