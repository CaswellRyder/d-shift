"""Frozen candidate evaluation on repeated IMG development scenes, not qualification."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.shared_quality_inference import infer
from scripts.proposal_quality_data import checked, load_parent
from scripts.shared_quality_data import run_path
from scripts.audit_scale_balanced_search import checked_panels
from scripts.balloon_dense_runtime import prepare_input, DenseProposals
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_clutter_feature_correction import LOCALIZER, LOCALIZER_SHA
from scripts.evaluate_current_confident_containment import checked_reference
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_predictor import classify_candidates
from scripts.research_confident_containment import select
from scripts.audit_source_aware_balloon import with_predictions, aggregate
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.summarize_shared_quality import meets94

PROTOCOL = Path('configs/shared-quality-development-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/shared-quality-development-20261011.json')
RAW = Path('runs/shared-quality-development-scenes-20261011.json')


def load_frozen(p):
    import keras
    for name in ('training_audit', 'selection', 'baseline'):
        if sha256(p[name]) != p[name+'_sha256']:
            raise ValueError('Prospectively frozen evidence changed')
    selection, audit = read_json(p['selection']), read_json(p['training_audit'])
    if selection['selected_for_next_evaluation'] != p['candidates'] or p['candidates'] != ['iou47_frozen', 'iou48_frozen']:
        raise ValueError('Candidate selection changed')
    if not selection['matched_head_gates']['iou_frozen'] or sha256(audit['raw']['path']) != audit['raw']['sha256']:
        raise ValueError('Training gate or raw receipt changed')
    for path, digest in audit['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Training source changed')
    _, parents, _, _ = checked()
    candidates, identities = {}, {}
    for seed in (47, 48):
        _, extractor, parent_root = load_parent(parents, seed)
        record, root = audit['training'][f'iou{seed}'], run_path('iou', seed)
        if read_json(root/'status.json')['state'] != 'complete' or sha256(root/'report.json') != record['report_sha256']:
            raise ValueError('Frozen head run changed')
        for name, digest in record['artifacts_sha256'].items():
            if sha256(root/name) != digest:
                raise ValueError('Head artifacts changed')
        head = keras.models.load_model(root/'quality.keras', compile=False)
        with np.load(root/'training.npz', allow_pickle=False) as training:
            mean, scale = training['mean'].copy(), training['scale'].copy()
            actual = head(training['normalized'], training=False).numpy()
        np.testing.assert_allclose(actual, np.load(root/'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0)
        c = parents['classifiers'][str(seed)]
        if sha256(c['path']) != c['sha256']:
            raise ValueError('Classifier changed')
        candidates[seed] = dict(extractor=extractor, classifier=keras.models.load_model(c['path'], compile=False),
                                head=head, mean=mean, scale=scale)
        identities[str(seed)] = dict(localizer=dict(path=str(parent_root/'proposal.keras'), sha256=sha256(parent_root/'proposal.keras')),
            classifier=c, head=dict(path=str(root/'quality.keras'), sha256=sha256(root/'quality.keras')),
            normalization=dict(path=str(root/'training.npz'), sha256=sha256(root/'training.npz')))
    return candidates, identities, audit


def infer_panel(sources, candidates, native):
    x = np.concatenate([prepare_input(rgb) for _, rgb in sources])
    outputs = {s: [v.numpy() for v in c['extractor'](x, training=False)] for s, c in candidates.items()}
    rows = {f'{arm}{seed}': [] for arm in ('parent', 'quality') for seed in candidates}
    bases = []
    for index, (frame, rgb) in enumerate(sources):
        base = search(rgb, native, True)
        bases.append(base)
        for seed, c in candidates.items():
            values, maps = outputs[seed]
            parent, quality, count = infer(rgb, base, values[index], maps[index], c['classifier'], c['head'], c['mean'], c['scale'])
            for arm, ds in [('parent', parent), ('quality', quality)]:
                result = with_predictions(frame, ds)
                result.update(neural_calls=len(base), full_frame_neural_calls=1,
                    quality_head_calls=int(count > 0) if arm == 'quality' else 0, quality_rows=count if arm == 'quality' else 0,
                    vertical_flip=frame.get('vertical_flip', False))
                rows[f'{arm}{seed}'].append(result)
    return {name: aggregate(fs) for name, fs in rows.items()}, bases


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve frozen evaluation')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    p = read_json(PROTOCOL)
    candidates, identities, train_audit = load_frozen(p)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native reducer changed')
    saved_train = read_json(train_audit['raw']['path'])
    train_parity = {}
    for panel, sources in checked_panels().items():
        results, _ = infer_panel(sources, candidates, native)
        for seed in candidates:
            for arm, old_arm in [('parent', 'parent'), ('quality', 'iou')]:
                for actual, old in zip(results[f'{arm}{seed}']['frames'], saved_train[panel][f'{old_arm}{seed}_frozen']['frames'], strict=True):
                    check_replayed_parent(actual, old)
                    for a, b in zip(actual['detections'], old['detections'], strict=True):
                        if ('box_quality' in a) != ('box_quality' in b) or abs(a.get('box_quality', 0)-b.get('box_quality', 0)) > 1e-5:
                            raise ValueError('TRAIN quality decisions changed')
        train_parity[panel] = len(sources)
    print(dict(fresh_train_search_and_candidate_parity=train_parity), flush=True)
    checked_reference()
    baseline = read_json(p['baseline'])
    if sha256(baseline['raw']['path']) != baseline['raw']['sha256']:
        raise ValueError('Incumbent reference changed')
    previous = read_json(baseline['raw']['path'])
    if sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Incumbent model changed')
    predictor, localizer = Predictor(MODEL, allow_unvalidated=True), DenseProposals(LOCALIZER)
    raw, panels, inputs = {}, {}, {}
    seen = set()
    for panel, count in p['panels'].items():
        queue, review_path = Path('data')/panel, Path('configs')/(panel+'.json')
        rows, review = reviewed_frames(queue, review_path)
        if len(rows) != count or review['threshold'] != .8 or review['matching_iou'] != .5:
            raise ValueError('Development panel coverage or operating point changed')
        pixels = {r['source']: (r, rgb) for r, rgb in rows}
        if seen & set(pixels) or any('IMG_' not in source for source in pixels):
            raise ValueError('Duplicate development source or wrong reserved family')
        seen |= set(pixels)
        sources = []
        for old in previous[panel]['frames']:
            row, rgb = pixels[old['source']]
            if row['sha256'] != old['frame_sha256'] or row['provisional_truth'] != old['truth']:
                raise ValueError('Development source or orientation changed')
            sources.append((dict(id=row['id'], source=row['source'], frame_sha256=row['sha256'],
                                 truth=row['provisional_truth'], vertical_flip=row['vertical_flip']), rgb))
        raw[panel], bases = infer_panel(sources, candidates, native)
        incumbent_frames = []
        for (frame, rgb), base, old in zip(sources, bases, previous[panel]['frames'], strict=True):
            localizer.predict(rgb)
            grid = localizer.interpreter.get_tensor(localizer.output['index'])[0]
            proposed = supplement(base, decode(grid, True))
            ds = select(classify_candidates(predictor, rgb, proposed))
            result = with_predictions(frame, ds)
            result.update(neural_calls=len(proposed), full_frame_neural_calls=1, quality_head_calls=0,
                          quality_rows=0, vertical_flip=frame['vertical_flip'])
            check_replayed_parent(result, old)
            incumbent_frames.append(result)
        raw[panel]['incumbent'] = aggregate(incumbent_frames)
        panels[panel] = {name: screen(result, raw[panel]['incumbent']) for name, result in raw[panel].items()}
        for seed in candidates:
            panels[panel][f'quality{seed}']['quality_effect'] = difference(raw[panel][f'quality{seed}'], raw[panel][f'parent{seed}'])
        for path in (queue/'review.json', review_path):
            inputs[str(path)] = sha256(path)
        print(dict(panel=panel, metrics={name: result['metrics'] for name, result in raw[panel].items()}), flush=True)
    write_json(RAW, raw)
    helpers = [__file__, PROTOCOL, 'scripts/shared_quality_inference.py', 'scripts/shared_proposal_quality.py',
        'scripts/evaluate_reviewed_balloon_scenes.py', 'scripts/research_blue_bright_search.py',
        'src/dtr/native_balloon_regions.py', 'scripts/research_predictor.py', 'scripts/audit_source_aware_balloon.py',
        'scripts/audit_balanced_localizer.py', 'scripts/diagnose_clutter_round3_feature.py',
        'scripts/diagnose_clutter_stage_attribution.py', 'scripts/summarize_shared_quality.py']
    write_json(OUTPUT, dict(inputs_sha256={**train_audit['inputs_sha256'], **inputs, **{str(path): sha256(path) for path in helpers}},
        protocol_sha256=sha256(PROTOCOL), candidates=identities, panels=panels, train_parity=train_parity,
        baseline_sha256=sha256(p['baseline']), native_sha256=sha256(native.path),
        raw=dict(path=str(RAW), sha256=sha256(RAW)),
        meets94_every_development_panel={f'quality{s}': all(meets94(panel[f'quality{s}']['metrics']) for panel in panels.values()) for s in candidates},
        scope=p['scope'], training_performed=False, development_evaluated=True, test_evaluated=False,
        independent_recording_sessions_verified=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    evaluate()
