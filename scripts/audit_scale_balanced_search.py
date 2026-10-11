"""Paired full-search TRAIN screen of one fixed color/scale allocation policy."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_scene_decision_retention import LOCALIZER, LOCALIZER_SHA
from scripts.balloon_dense_runtime import DenseProposals
from scripts.research_blue_bright_search import search
from scripts.research_balloon_search import select as original_select
from scripts.scale_balanced_balloon_search import pool, balanced_select
from scripts.research_blue_peak_blend import decode
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_predictor import classify_candidates
from scripts.research_confident_containment import select
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.balloon_scene_constraints import RAW as OLD, RAW_SHA
from scripts.audit_window1_scene_search import OUTPUT as WINDOW1, REVIEW as REVIEW1
from scripts.audit_window2_scene_search import OUTPUT as WINDOW2
from scripts.window2_scene_labels import checked_frames, REVIEW as REVIEW2
from scripts.prepare_openimages_balloon_review import checked_path

PROTOCOL = Path('configs/scale-balanced-search-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/scale-balanced-search-20261011.json')
RAW = Path('runs/scale-balanced-search-20261011.json')


def checked_panels():
    if sha256(OLD) != RAW_SHA:
        raise ValueError('Original TRAIN reference changed')
    old = read_json(OLD)['training']['parent']['frames']
    references = {f['source']: f for f in old}
    original = []
    # Stop once all original64 are collected; do not scan unrelated public crops.
    for source, rgb in source_frames():
        if source['origin'] != 'complete_fit' or 'IMG_' in source['source']:
            raise ValueError('Unexpected training source')
        f = references[source['source']]
        if source['source_file_sha256'] != f['frame_sha256'] or source['truth'] != f['truth']:
            raise ValueError('Original frame identity changed')
        original.append((f, rgb))
        if len(original) == 64:
            break
    if len(original) != 64:
        raise ValueError('Incomplete original TRAIN panel')
    report = read_json(WINDOW1)
    path = report['raw']['path']
    if sha256(path) != report['raw']['sha256'] or sha256(REVIEW1) != report['inputs_sha256'][str(REVIEW1)]:
        raise ValueError('Window1 baseline changed')
    review = read_json(REVIEW1)
    root = Path(review['queue'])
    if sha256(root/'review.json') != review['queue_sha256']:
        raise ValueError('Window1 source queue changed')
    queue = read_json(root/'review.json')
    window1 = []
    for f in read_json(path)['frames']:
        source = queue['frames'][f['id']]
        truth = [dict(annotation_id=i, label=review['labels'][str(i)], box=queue['samples'][i]['box'])
                 for i in review['complete_red_blue_frames'][str(f['id'])]]
        if (truth != f['truth'] or source['sha256'] != f['frame_sha256'] or
                'openimages/train/'+source['image_id'] != f['source']):
            raise ValueError('Window1 identity changed')
        with Image.open(checked_path(root, source['path'], source['sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        window1.append((f, rgb))
    report = read_json(WINDOW2)
    path = report['raw']['path']
    if sha256(path) != report['raw']['sha256'] or sha256(REVIEW2) != report['inputs_sha256'][str(REVIEW2)]:
        raise ValueError('Window2 baseline changed')
    references = {f['id']: f for f in read_json(path)['parent']['frames']}
    window2 = []
    for frame, rgb in checked_frames():
        f = references[frame['id']]
        if any(frame[k] != f[k] for k in frame):
            raise ValueError('Window2 frame identity changed')
        window2.append((f, rgb))
    if len(window1) != 6 or len(window2) != 12:
        raise ValueError('Incomplete public TRAIN panels')
    return dict(original=original, window1=window1, window2=window2)


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve paired search evidence')
    protocol = read_json(PROTOCOL)
    if [protocol[k] for k in ('small_max_side', 'quota_per_color_scale', 'total_limit', 'duplicate_iou',
                             'classifier_threshold', 'matching_iou')] != [32, 3, 12, .65, .8, .5]:
        raise ValueError('Fixed experiment changed')
    if sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Frozen models changed')
    classifier, localizer = Predictor(MODEL, allow_unvalidated=True), DenseProposals(LOCALIZER)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if classifier.metadata['threshold'] != .8 or read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Runtime policy or native source changed')
    panels = checked_panels()
    raw, summary = {}, {}
    for name, panel_frames in panels.items():
        arms = dict(parent=[], balanced=[])
        for frame, rgb in panel_frames:
            candidates = pool(rgb, native)
            base = original_select(candidates)
            if base != search(rgb, native, True):
                raise ValueError('Search pool does not reproduce baseline')
            localizer.predict(rgb)
            learned = decode(localizer.interpreter.get_tensor(localizer.output['index'])[0], True)
            for arm, selected in (('parent', base), ('balanced', balanced_select(candidates))):
                proposals = supplement(selected, learned)
                if len(proposals) > 12:
                    raise ValueError('Exceeded existing per-frame neural budget')
                found = select(classify_candidates(classifier, rgb, proposals))
                row = dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                    detections=found, counts=detection_counts(frame['truth'], found), neural_calls=len(proposals))
                if arm == 'parent':
                    check_replayed_parent(row, frame)
                arms[arm].append(row)
        results = {arm: dict(frames=frames, metrics=metrics({c: {k: sum(f['counts'][c][k] for f in frames)
            for k in ('tp', 'fp', 'fn')} for c in LABELS})) for arm, frames in arms.items()}
        effect = difference(results['balanced'], results['parent'])
        passed = not effect['lost_targets'] and all(effect['count_delta'][c]['fp'] <= 0 for c in LABELS)
        raw[name] = results
        summary[name] = dict(metrics={a: checked_metrics(r) for a, r in results.items()}, effect=effect,
            no_regression=bool(passed), neural_calls={a: sum(f['neural_calls'] for f in fs) for a, fs in arms.items()},
            maximum_calls={a: max(f['neural_calls'] for f in fs) for a, fs in arms.items()})
        print(dict(panel=name, **summary[name]), flush=True)
    eligible = all(s['no_regression'] for s in summary.values()) and any(s['effect']['gained_targets'] for s in summary.values())
    write_json(RAW, raw)
    inputs = [PROTOCOL, REVIEW1, REVIEW2, WINDOW1, WINDOW2, OLD, MODEL, LOCALIZER,
        __file__, 'scripts/scale_balanced_balloon_search.py', 'scripts/research_balloon_search.py',
        'scripts/research_blue_bright_search.py', 'scripts/research_blue_peak_blend.py',
        'scripts/research_confident_containment.py', 'scripts/research_predictor.py',
        'scripts/audit_balloon_dense_proposals.py', 'scripts/diagnose_clutter_stage_attribution.py', native.path]
    write_json(OUTPUT, dict(panels=summary, eligible_for_separate_development=bool(eligible),
        inputs_sha256={str(p): sha256(p) for p in inputs}, raw=dict(path=str(RAW), sha256=sha256(RAW)),
        decision='Proceed only to separate DEV investigation.' if eligible else 'Reject scale quota policy: TRAIN retention/false-positive gate failed.',
        scope='One fixed candidate allocation tested on82 already-admitted TRAIN photos. Parent search reproduced exactly, both arms use same weights, threshold and12maximumcalls. Actual calls may differ; no latency or held-out accuracy claim.',
        training_performed=False, development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
