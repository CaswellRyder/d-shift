"""Reload source-resolved negative retraining and evaluate fixed TRAIN policies."""
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.source_resolved_negative_data import tensors, matched, run_path, PROTOCOL
from scripts.public_scene_localizer_data import tensor_hash
from scripts.clutter_round3_feature_data import targets, loss_function
from scripts.train_batch3_retraining import correct
from scripts.audit_scale_balanced_search import checked_panels
from scripts.audit_direct_hybrid_balloon import checked_inputs as hybrid_inputs, raw_direct
from scripts.audit_direct_hybrid_balloon import OUTPUT as HYBRID
from scripts.audit_source_aware_balloon import aggregate, with_predictions
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_stage_attribution import crops, difference
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.direct_hybrid_balloon import merge

OUTPUT = Path('research/balloon-red-blue-20261008/source-resolved-negative-20261011.json')
RAW = Path('runs/source-resolved-negative-scenes-20261011.json')


def eligible_policies(summary):
    if set(summary) != {'original', 'window1', 'window2'}:
        raise ValueError('Incomplete TRAIN panels')
    return {policy: all(all(p[f'corrected{s}_{policy}']['no_regression'] for p in summary.values()) and
        any(p[f'corrected{s}_{policy}']['effect']['gained_targets'] or
            any(v['fp'] < 0 for v in p[f'corrected{s}_{policy}']['effect']['count_delta'].values())
            for p in summary.values()) for s in (47, 48)) for policy in ('current', 'pooled', 'base_first')}


def verify(arm, seed, data, parent):
    import keras
    import tensorflow as tf
    root = run_path(arm, seed)
    prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
    if (read_json(root/'status.json')['state'] != 'complete' or prov['arm'] != arm or prov['seed'] != seed
            or prov['settings'] != data['settings'] or prov['parent'] != data['parent']
            or prov['sources_sha256'] != data['sources_sha256'] or prov['fresh_optimizer'] is not True
            or prov['fixed_final_epoch'] is not True or prov['parameters'] != 7763):
        raise ValueError('Training identity changed')
    for k in ('corrections', 'scene_ids', 'reviewed_negatives', 'existing_pixel_matches', 'positive_records', 'duplicate_public_inputs'):
        if prov[k] != data[k]:
            raise ValueError('Supervision metadata changed')
    for k in ('development_evaluated', 'test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'):
        if prov[k] is not False:
            raise ValueError('Require TRAIN-only checkpoint')
    if any(sha256(path) != h for path, h in data['sources_sha256'].items()):
        raise ValueError('Training dependency changed')
    for name, digest in report['artifacts_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Training artifact changed')
    for key, digest in prov['tensor_sha256'].items():
        if tensor_hash(data[key]) != digest:
            raise ValueError('Training tensor changed')
    before = parent(data['x'], training=False).numpy()
    target = targets(data['y'], before, data['preserve'], data['scene_mask'])
    if tensor_hash(target) != prov['target_sha256']:
        raise ValueError('Teacher targets changed')
    p = data['settings']
    loss = loss_function(p['scene_kl_weight'], p['scene_centered_logit_weight'])
    model = keras.models.load_model(root/'student.keras', custom_objects={'loss': loss})
    if (model.get_config() != parent.get_config() or model.count_params() != 7763
            or int(model.optimizer.iterations) != 13440 or report['optimizer_steps'] != 13440):
        raise ValueError('Architecture or restored optimizer steps changed')
    with (root/'epochs.csv').open() as handle:
        trace = list(csv.DictReader(handle))
    if [int(r['epoch']) for r in trace] != list(range(80)) or not all(np.isfinite(float(r['loss'])) for r in trace):
        raise ValueError('Incomplete training trace')
    after = model(data['x'], training=False).numpy()
    negatives = model(data['negative_pixels'], training=False).numpy()
    probes = model(data['probes'], training=False).numpy()
    with np.load(root/'predictions.npz', allow_pickle=False) as stored:
        for key, value in dict(before=before, after=after, negatives=negatives, probes=probes).items():
            np.testing.assert_allclose(stored[key], value, atol=1e-5, rtol=0)
    fit_loss = float(np.mean(loss(tf.constant(target), tf.constant(after)).numpy()*data['weights']))
    if not np.isclose(fit_loss, report['fit_loss'], atol=1e-5, rtol=0):
        raise ValueError('Fitting loss does not reproduce')
    old_before = correct(before[:2074], data['y'][:2074])
    old_after = correct(after[:2074], data['y'][:2074])
    def false_accepts(logits):
        probability = softmax(logits)
        return int(((probability.argmax(1) > 0) & (probability.max(1) >= .8)).sum())
    evidence = dict(run=str(root), artifacts_sha256={**report['artifacts_sha256'], 'report.json': sha256(root/'report.json')},
        fit_loss=fit_loss, optimizer_steps=13440, negatives_false_accepted=false_accepts(negatives),
        negatives_probabilities=softmax(negatives).tolist(),
        original_correct_before=int(old_before.sum()), original_correct_after=int(old_after.sum()),
        lost_original_rows=np.flatnonzero(old_before & ~old_after).tolist(),
        prior_negative_false_accepts=[false_accepts(parent(data['probes'], training=False).numpy()), false_accepts(probes)])
    return model, evidence


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve correction experiment')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    models, training = {}, {}
    for seed in (47, 48):
        arms = {a: tensors(a, seed) for a in ('control', 'corrected')}
        matched(arms['control'], arms['corrected'])
        parent = keras.models.load_model(arms['control']['parent']['path'], compile=False)
        models[f'parent{seed}'] = parent
        for arm, data in arms.items():
            models[f'{arm}{seed}'], training[f'{arm}{seed}'] = verify(arm, seed, data, parent)
            print(dict(verified=f'{arm}{seed}', negative_fp=training[f'{arm}{seed}']['negatives_false_accepted'],
                       old_lost=len(training[f'{arm}{seed}']['lost_original_rows'])), flush=True)
    spec = read_json(PROTOCOL)
    parent_audit = read_json(spec['parent_audit'])
    if sha256(spec['parent_audit']) != spec['parent_audit_sha256'] or sha256(parent_audit['raw']['path']) != parent_audit['raw']['sha256']:
        raise ValueError('Previous correction reference changed')
    parent_results = read_json(parent_audit['raw']['path'])
    _, refs, _, _ = hybrid_inputs()
    hybrid = read_json(HYBRID)
    for path, digest in {**hybrid['inputs_sha256'], **hybrid['models_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Hybrid source changed')
    if sha256(hybrid['raw']['path']) != hybrid['raw']['sha256']:
        raise ValueError('Frozen hybrid predictions changed')
    previous = read_json(hybrid['raw']['path'])
    original = read_json(refs['joint']['raw']['path'])
    raw, summary = {}, {}
    for panel, sources in checked_panels().items():
        pixels = {f['source']: (f['frame_sha256'], f['truth'], rgb) for f, rgb in sources}
        current_frames = original[panel]['parent_parent47_current']['frames']
        x_current = crops(current_frames, pixels)
        base_frames = previous[panel]['base_parent']['frames']
        x_base = crops(base_frames, pixels)
        results = {}
        for name, model in models.items():
            def predict(x):
                return softmax(np.concatenate([model(x[i:i+64], training=False).numpy()
                                               for i in range(0, len(x), 64)]))
            current = replay(current_frames, predict(x_current))
            base = replay(base_frames, predict(x_base))
            seed = name[-2:]
            if name.startswith('parent'):
                reference = parent_results[panel][f'corrected{seed}_current']
                for actual, saved in zip(current['frames'], reference['frames'], strict=True):
                    check_replayed_parent(actual, saved)
            results[f'{name}_current'] = current
            for policy in ('pooled', 'base_first'):
                frames = []
                for i, frame in enumerate(base['frames']):
                    old = previous[panel][f'balanced_admit{seed}_{policy}']['frames'][i]
                    if any(frame[k] != old[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                        raise ValueError('Frame alignment changed')
                    direct = raw_direct(dict(detections=[d for d in old['detections'] if d['proposal_origin'] == 'learned']))
                    selected = merge(raw_direct(frame), direct, policy)
                    row = with_predictions(frame, selected)
                    row.update(neural_calls=len(frame['detections']), full_frame_neural_calls=1)
                    if name.startswith('parent'):
                        check_replayed_parent(row, parent_results[panel][f'corrected{seed}_{policy}']['frames'][i])
                    frames.append(row)
                results[f'{name}_{policy}'] = aggregate(frames)
        raw[panel] = results
        summary[panel] = {name: screen(result, original[panel]['parent_parent47_current'])
                          for name, result in results.items()}
        for seed in (47, 48):
            for policy in ('current', 'pooled', 'base_first'):
                name = f'corrected{seed}_{policy}'
                summary[panel][name]['matched_control_effect'] = difference(results[name], results[f'control{seed}_{policy}'])
        print(dict(panel=panel, corrected={n: r['metrics'] for n, r in summary[panel].items()
                                          if n.startswith('corrected')}), flush=True)
    eligible = eligible_policies(summary)
    write_json(RAW, raw)
    paths = [PROTOCOL, __file__, HYBRID, spec['parent_audit'], 'scripts/positive_proposal_supervision.py', 'scripts/source_resolved_negative_data.py', 'scripts/prepare_source_resolved_negatives.py', 'configs/source-resolved-negative-review-20261011.json',
        'scripts/clutter_round3_feature_data.py', 'scripts/direct_hybrid_balloon.py',
        'scripts/audit_balanced_localizer.py', 'scripts/diagnose_clutter_round3_feature.py',
        'scripts/diagnose_clutter_stage_attribution.py']
    write_json(OUTPUT, dict(training=training, panels=summary,
        eligible_for_separate_export_development=eligible, raw=dict(path=str(RAW), sha256=sha256(RAW)),
        inputs_sha256={str(path): sha256(path) for path in paths},
        decision='Separate export/DEV investigation permitted only for passing paired policy.' if any(eligible.values())
                 else 'Reject source-resolved-negative trial: paired scene-retention/improvement gate failed.',
        scope='Three source-resolution reviewed non-target crops, four matched retrainings, restored optimizer/checkpoint/tensor verification,18runtime variants on82TRAIN scenes. Frozen geometry, classifier thresholds and both direct merge policies. No held-out evaluation or Pi execution.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=eligible), flush=True)


if __name__ == '__main__':
    audit()
