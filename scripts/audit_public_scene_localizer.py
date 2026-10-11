"""Verify matched localizer training and screen fixed-budget detections on TRAIN only."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.public_scene_localizer_data import tensors, matched, tensor_hash, run_path, dependencies, PROTOCOL
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_scene_decision_retention import LOCALIZER, LOCALIZER_SHA
from scripts.balloon_dense_runtime import DenseProposals, prepare_input
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.audit_balloon_dense_proposals import supplement
from scripts.reserved_dense_balloon_search import reserve
from scripts.research_predictor import classify_candidates
from scripts.research_confident_containment import select
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_pipeline_errors import analyze_frame

OUTPUT = Path('research/balloon-red-blue-20261008/public-scene-localizer-20261011.json')
RAW = Path('runs/public-scene-localizer-scenes-20261011.json')


def verify(arm, seed, data, parent):
    import keras
    import tensorflow as tf
    from scripts.balloon_dense_iou_loss import giou_loss
    root = run_path(arm, seed)
    prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
    if (read_json(root/'status.json')['state'] != 'complete' or prov['arm'] != arm or prov['seed'] != seed
            or prov['settings'] != data['settings'] or prov['inputs_sha256'] != tensor_hash(data['x'])
            or prov['targets_sha256'] != tensor_hash(data['y']) or prov['sources_sha256'] != dependencies()
            or prov['fit_frames'] != 139 or prov['excluded_original_holdout_frames'] != 16
            or prov['fresh_optimizer'] is not True or prov['fixed_final_checkpoint'] is not True
            or report['optimizer_steps'] != 3600):
        raise ValueError('Training identity or schedule changed')
    for key in ('original_fit_indices', 'original_groups', 'negative_frames', 'additional_frames', 'original_folds'):
        if prov[key] != data[key]:
            raise ValueError('Training sources or splits changed')
    if any(prov[k] is not False for k in ('original_holdout_evaluated', 'development_evaluated',
                                         'test_evaluated', 'deployment_approved', 'qualification_achieved')):
        raise ValueError('Require TRAIN-only experiment')
    for key, name in (('model_sha256', 'proposal.keras'), ('predictions_sha256', 'predictions.npy'),
                      ('provenance_sha256', 'provenance.json')):
        if sha256(root/name) != report[key]:
            raise ValueError('Training artifact changed')
    with (root/'epochs.csv').open() as handle:
        trace = list(csv.DictReader(handle))
    if [int(r['epoch']) for r in trace] != list(range(200)) or not all(np.isfinite(float(r['loss'])) for r in trace):
        raise ValueError('Incomplete or nonfinite training trace')
    model = keras.models.load_model(root/'proposal.keras', compile=False)
    if model.count_params() != 2342 or model.get_config() != parent.get_config():
        raise ValueError('Model graph changed')
    actual = model(data['x'], training=False).numpy()
    stored = np.load(root/'predictions.npy', allow_pickle=False)
    np.testing.assert_allclose(actual, stored, atol=1e-5, rtol=0)
    fit_loss = float(giou_loss(tf.constant(data['y']), tf.constant(actual)))
    if not np.isclose(fit_loss, report['fit_loss'], atol=1e-5, rtol=0):
        raise ValueError('Saved fitting loss does not reproduce')
    evidence = dict(run=str(root), fit_loss=fit_loss, optimizer_steps=3600, parameters=2342,
        artifacts_sha256={name: sha256(root/name) for name in
            ('proposal.keras', 'predictions.npy', 'provenance.json', 'report.json', 'epochs.csv')})
    return model, evidence


def screen(result, parent):
    effect = difference(result, parent)
    details = [analyze_frame(f) for f in result['frames']]
    return dict(metrics=checked_metrics(result), effect=effect,
        no_regression=not effect['lost_targets'] and all(effect['count_delta'][c]['fp'] <= 0 for c in LABELS),
        geometry_oracle=sum(f['matched_upper_bounds']['geometry_oracle'] for f in details),
        errors={k: dict(Counter(r['reason'] for f in details for r in f[k]))
                for k in ('false_negatives', 'false_positives')},
        neural_calls=sum(f['neural_calls'] for f in result['frames']),
        maximum_calls=max(f['neural_calls'] for f in result['frames']))


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve fixed-budget localizer evidence')
    arms = {a: tensors(a) for a in ('control', 'scenes')}
    matched(arms['control'], arms['scenes'])
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(arms['control']['settings']['parent'], compile=False)
    models, training = {}, {}
    for arm, data in arms.items():
        for seed in (47, 48):
            models[f'{arm}{seed}'], training[f'{arm}{seed}'] = verify(arm, seed, data, parent)
    if sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Frozen deployed-format research models changed')
    classifier, localizer = Predictor(MODEL, allow_unvalidated=True), DenseProposals(LOCALIZER)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if classifier.metadata['threshold'] != .8 or read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Operating point or native source changed')
    raw, panels = {}, {}
    for panel, frames in arms['control']['panels'].items():
        x = np.concatenate([prepare_input(rgb) for _, rgb in frames])
        predictions = {name: m(x, training=False).numpy() for name, m in models.items()}
        keras_parent = parent(x, training=False).numpy()
        found = {f'{name}_{policy}': [] for name in ('parent', *models) for policy in ('current', 'reserved')}
        for index, (frame, rgb) in enumerate(frames):
            base = search(rgb, native, True)
            localizer.predict(rgb)
            reference = localizer.interpreter.get_tensor(localizer.output['index'])[0]
            np.testing.assert_allclose(reference, keras_parent[index], atol=1e-5, rtol=0)
            for name in ('parent', *models):
                values = reference if name == 'parent' else predictions[name][index]
                learned = decode(values, True)
                for policy, fn in (('current', supplement), ('reserved', reserve)):
                    proposals = fn(base, learned)
                    if len(proposals) > 12:
                        raise ValueError('Exceeded neural candidate budget')
                    detections = select(classify_candidates(classifier, rgb, proposals))
                    row = dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                        detections=detections, counts=detection_counts(frame['truth'], detections), neural_calls=len(proposals))
                    if name == 'parent' and policy == 'current':
                        check_replayed_parent(row, frame)
                    found[f'{name}_{policy}'].append(row)
        results = {name: dict(frames=fs, metrics=metrics({c: {k: sum(f['counts'][c][k] for f in fs)
            for k in ('tp', 'fp', 'fn')} for c in LABELS})) for name, fs in found.items()}
        raw[panel] = results
        panels[panel] = {name: screen(r, results['parent_current']) for name, r in results.items()}
        for seed in (47, 48):
            for policy in ('current', 'reserved'):
                name = f'scenes{seed}_{policy}'
                panels[panel][name]['against_matched_control'] = difference(results[name], results[f'control{seed}_{policy}'])
        print(dict(panel=panel, metrics={k: v['metrics'] for k, v in panels[panel].items()}), flush=True)
    eligible = {policy: all(all(p[f'scenes{seed}_{policy}']['no_regression'] for p in panels.values()) and
        any(p[f'scenes{seed}_{policy}']['effect']['gained_targets'] for p in panels.values())
        for seed in (47, 48)) for policy in ('current', 'reserved')}
    write_json(RAW, raw)
    helpers = [__file__, 'scripts/reserved_dense_balloon_search.py', 'scripts/research_blue_peak_blend.py',
        'scripts/research_blue_bright_search.py', 'scripts/audit_balloon_dense_proposals.py',
        'scripts/research_confident_containment.py', 'scripts/research_predictor.py',
        'scripts/summarize_public_balloon_trial.py', 'scripts/diagnose_clutter_stage_attribution.py',
        'scripts/audit_balloon_pipeline_errors.py', MODEL, LOCALIZER, native.path, PROTOCOL]
    write_json(OUTPUT, dict(training=training, panels=panels, eligible_for_separate_export_development=eligible,
        raw=dict(path=str(RAW), sha256=sha256(RAW)),
        inputs_sha256={**dependencies(), **{str(p): sha256(p) for p in helpers}},
        decision='Separate export/DEV investigation permitted only for passing paired policy.' if any(eligible.values()) else
                 'Reject: neither fixed-budget policy passes paired TRAIN retention and recovery.',
        scope='Both seeds and matched supervision controls;82 previously admitted TRAIN scenes. Frozen classifier, unchanged2342parameter localizer graph, maximum12classifications per frame. Keras candidate predictions: export parity and Pi latency untested. Assistant scene labels are not independent qualification.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
