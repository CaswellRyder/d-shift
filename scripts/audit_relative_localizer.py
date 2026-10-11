"""Matched relative-box objective and fixed decoder policies on TRAIN scenes only."""
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.public_scene_localizer_data import tensors, tensor_hash
from scripts.train_relative_localizer import run_path, settings, sources, PROTOCOL
from scripts.audit_public_scene_localizer import verify as verify_control, screen
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_scene_decision_retention import LOCALIZER, LOCALIZER_SHA
from scripts.balloon_dense_runtime import DenseProposals, prepare_input
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.box_nms_dense_decoder import decode as box_decode
from scripts.audit_balloon_dense_proposals import supplement
from scripts.reserved_dense_balloon_search import reserve
from scripts.research_predictor import classify_candidates
from scripts.research_confident_containment import select
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference

EVALUATION = Path('configs/relative-localizer-evaluation-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/relative-localizer-20261011.json')
RAW = Path('runs/relative-localizer-scenes-20261011.json')


def verify_relative(seed, data, parent, control):
    import keras
    import tensorflow as tf
    from scripts.balloon_relative_box_loss import relative_loss, relative_penalty
    from scripts.balloon_dense_iou_loss import giou_loss
    root = run_path(seed)
    prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
    if (read_json(root/'status.json')['state'] != 'complete' or prov['seed'] != seed
            or prov['settings'] != settings() or prov['base_settings'] != data['settings']
            or prov['sources_sha256'] != sources() or prov['inputs_sha256'] != tensor_hash(data['x'])
            or prov['targets_sha256'] != tensor_hash(data['y']) or prov['verified_control'] != control
            or prov['fresh_optimizer'] is not True or prov['fixed_final_checkpoint'] is not True
            or report['optimizer_steps'] != 3600 or prov['parameters'] != 2342):
        raise ValueError('Training objective, matched control or schedule changed')
    if any(prov[k] is not False for k in ('original_holdout_evaluated', 'development_evaluated',
            'test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved')):
        raise ValueError('TRAIN-only scope changed')
    for key, filename in (('model_sha256', 'proposal.keras'), ('predictions_sha256', 'predictions.npy'),
                           ('provenance_sha256', 'provenance.json')):
        if sha256(root/filename) != report[key]:
            raise ValueError('Training artifact changed')
    with (root/'epochs.csv').open() as handle:
        trace = list(csv.DictReader(handle))
    if [int(r['epoch']) for r in trace] != list(range(200)) or not all(np.isfinite(float(r['loss'])) for r in trace):
        raise ValueError('Missing or nonfinite epochs')
    model = keras.models.load_model(root/'proposal.keras', compile=False)
    if model.count_params() != 2342 or model.get_config() != parent.get_config():
        raise ValueError('Inference graph changed')
    actual = model(data['x'], training=False).numpy()
    np.testing.assert_allclose(actual, np.load(root/'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0)
    for key, fn in (('fit_loss', relative_loss), ('reference_giou_loss', giou_loss), ('relative_penalty', relative_penalty)):
        measured = float(fn(tf.constant(data['y']), tf.constant(actual)))
        if not np.isclose(measured, report[key], atol=1e-5, rtol=0):
            raise ValueError('Objective values do not reproduce')
    return model, dict(run=str(root), **report, parameters=2342,
        artifacts_sha256={f: sha256(root/f) for f in ('proposal.keras', 'predictions.npy',
            'provenance.json', 'report.json', 'epochs.csv')})


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve paired evidence')
    protocol = read_json(EVALUATION)
    if ([protocol[k] for k in ('classifier_threshold', 'matching_iou', 'proposal_heat_threshold',
            'box_nms_iou', 'total_limit')] != [.8, .5, .1, .65, 12]
            or protocol['variants'] != ['current_peaks', 'reserved_peaks', 'reserved_boxes']):
        raise ValueError('Fixed evaluator changed')
    data = tensors('scenes')
    settings()
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(data['settings']['parent'], compile=False)
    models, training = {}, {}
    for seed in (47, 48):
        models[f'control{seed}'], training[f'control{seed}'] = verify_control('scenes', seed, data, parent)
        models[f'relative{seed}'], training[f'relative{seed}'] = verify_relative(seed, data, parent, training[f'control{seed}'])
    if sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Frozen runtime-format reference changed')
    classifier, localizer = Predictor(MODEL, allow_unvalidated=True), DenseProposals(LOCALIZER)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if classifier.metadata['threshold'] != .8 or read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Operating point or native source changed')
    raw, panels = {}, {}
    for panel, frames in data['panels'].items():
        x = np.concatenate([prepare_input(rgb) for _, rgb in frames])
        predicted = {name: model(x, training=False).numpy() for name, model in models.items()}
        parent_predictions = parent(x, training=False).numpy()
        found = {f'{name}_{policy}': [] for name in ('parent', *models) for policy in protocol['variants']}
        for index, (frame, rgb) in enumerate(frames):
            base = search(rgb, native, True)
            localizer.predict(rgb)
            reference = localizer.interpreter.get_tensor(localizer.output['index'])[0]
            np.testing.assert_allclose(reference, parent_predictions[index], atol=1e-5, rtol=0)
            for name in ('parent', *models):
                values = reference if name == 'parent' else predicted[name][index]
                peaks = decode(values, True)
                for policy, proposals in (('current_peaks', supplement(base, peaks)),
                        ('reserved_peaks', reserve(base, peaks)), ('reserved_boxes', reserve(base, box_decode(values)))):
                    if len(proposals) > 12:
                        raise ValueError('Exceeded existing classification budget')
                    detections = select(classify_candidates(classifier, rgb, proposals))
                    row = dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                        detections=detections, counts=detection_counts(frame['truth'], detections), neural_calls=len(proposals))
                    if name == 'parent' and policy == 'current_peaks':
                        check_replayed_parent(row, frame)
                    found[f'{name}_{policy}'].append(row)
        results = {name: dict(frames=fs, metrics=metrics({c: {k: sum(f['counts'][c][k] for f in fs)
            for k in ('tp', 'fp', 'fn')} for c in LABELS})) for name, fs in found.items()}
        raw[panel] = results
        panels[panel] = {name: screen(result, results['parent_current_peaks']) for name, result in results.items()}
        for seed in (47, 48):
            for policy in protocol['variants']:
                name = f'relative{seed}_{policy}'
                panels[panel][name]['against_matched_control'] = difference(results[name], results[f'control{seed}_{policy}'])
            for arm in ('control', 'relative'):
                panels[panel][f'{arm}{seed}_reserved_boxes']['decoder_only_effect'] = difference(
                    results[f'{arm}{seed}_reserved_boxes'], results[f'{arm}{seed}_reserved_peaks'])
        print(dict(panel=panel, results={k: dict(metrics=v['metrics'], geometry=v['geometry_oracle'],
            lost=len(v['effect']['lost_targets'])) for k, v in panels[panel].items() if k.startswith('relative')}), flush=True)
    eligible = {policy: all(all(p[f'relative{seed}_{policy}']['no_regression'] for p in panels.values()) and
        any(p[f'relative{seed}_{policy}']['effect']['gained_targets'] for p in panels.values())
        for seed in (47, 48)) for policy in protocol['variants']}
    write_json(RAW, raw)
    helpers = [__file__, EVALUATION, PROTOCOL, 'scripts/box_nms_dense_decoder.py',
        'scripts/reserved_dense_balloon_search.py', 'scripts/research_blue_peak_blend.py',
        'scripts/research_blue_bright_search.py', 'scripts/research_predictor.py',
        'scripts/research_confident_containment.py', 'scripts/audit_balloon_dense_proposals.py',
        'scripts/summarize_public_balloon_trial.py', 'scripts/diagnose_clutter_stage_attribution.py',
        MODEL, LOCALIZER, native.path]
    write_json(OUTPUT, dict(training=training, panels=panels, eligible_for_separate_export_development=eligible,
        inputs_sha256={**sources(), **{str(p): sha256(p) for p in helpers}},
        raw=dict(path=str(RAW), sha256=sha256(RAW)),
        decision='Passing paired policy permits separate export/DEV investigation only.' if any(eligible.values()) else
                 'Reject relative-loss candidates: no fixed policy passes paired TRAIN detection checks.',
        scope='Objective-only matched retraining plus separated fixed decoder ablation on82TRAINscenes. No parameter/MAC growth, but actual candidate count and decoder cost can change. Keras predictions are not export parity or Pi performance evidence.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
