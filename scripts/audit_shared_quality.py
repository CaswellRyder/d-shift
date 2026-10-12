"""Matched shared-source quality objectives, frozen parent replay and TRAIN errors."""
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.shared_quality_data import build, dependencies, run_path, checked_previous
from scripts.shared_proposal_quality import network, feature_row, gate
from scripts.proposal_quality_data import checked, load_parent
from scripts.public_scene_localizer_data import tensor_hash
from scripts.audit_scale_balanced_search import checked_panels
from scripts.balloon_dense_runtime import prepare_input
from scripts.cached_balloon_decoder import decode
from scripts.audit_center_presence_localizer import classify, as_direct
from scripts.audit_direct_hybrid_balloon import raw_direct, checked_inputs as hybrid_inputs
from scripts.direct_hybrid_balloon import merge
from scripts.audit_source_aware_balloon import with_predictions, aggregate
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.summarize_proposal_quality_localizer import origins

OUTPUT = Path('research/balloon-red-blue-20261008/shared-proposal-quality-20261011.json')
RAW = Path('runs/shared-proposal-quality-scenes-20261011.json')
DECODERS = ('frozen', 'independent_neighbor')


def verify(arm, seed, data):
    import keras
    root = run_path(arm, seed)
    p, r = read_json(root/'provenance.json'), read_json(root/'report.json')
    if (read_json(root/'status.json')['state'] != 'complete' or p['arm'] != arm or p['seed'] != seed
            or p['sources_sha256'] != dependencies() or p['settings'] != data['settings']
            or p['parent_sha256'] != data['parent_sha256'] or p['frozen_parent'] is not True
            or p['fixed_final_epoch'] is not True):
        raise ValueError('Training identity changed')
    for key in ('identities', 'complete_sources', 'mser_sources', 'learned_rows'):
        if p[key] != data[key]:
            raise ValueError('Training examples or review scope changed')
    for name, digest in r['artifacts_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Checkpoint artifact changed')
    target = data['y'] if arm == 'iou' else (data['y'] >= .5).astype(np.float32)
    with np.load(root/'training.npz', allow_pickle=False) as saved:
        for key in ('x', 'y', 'normalized', 'mean', 'scale', 'weights'):
            np.testing.assert_array_equal(saved[key], data[key])
        np.testing.assert_array_equal(saved['target'], target)
    keras.utils.set_random_seed(seed)
    initialized = network()
    if [tensor_hash(v.numpy()) for v in initialized.weights] != p['initial_weights_sha256']:
        raise ValueError('Matched head initialization differs')
    model = keras.models.load_model(root/'quality.keras')
    if model.count_params() != 281 or int(model.optimizer.iterations) != p['optimizer_steps'] or r['optimizer_steps'] != p['optimizer_steps']:
        raise ValueError('Head or restored optimizer schedule changed')
    expected_loss = 'mean_squared_error' if arm == 'iou' else 'binary_crossentropy'
    if model.loss != expected_loss or p['loss'] != expected_loss:
        raise ValueError('Training objective changed')
    with (root/'epochs.csv').open() as handle:
        trace = list(csv.DictReader(handle))
    if [int(t['epoch']) for t in trace] != list(range(200)) or not all(np.isfinite(float(t['loss'])) for t in trace):
        raise ValueError('Training trace incomplete')
    pred = model(data['normalized'], training=False).numpy()
    np.testing.assert_allclose(pred, np.load(root/'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0)
    loss = float(model.evaluate(data['normalized'], target, sample_weight=data['weights'], verbose=0))
    if not np.isclose(loss, r['weighted_loss'], atol=1e-6, rtol=0):
        raise ValueError('Fitting loss does not reproduce')
    return model, dict(run=str(root), **r, report_sha256=sha256(root/'report.json'),
        initial_weights_sha256=p['initial_weights_sha256'], features_sha256=tensor_hash(data['x']),
        normalization_sha256=[tensor_hash(data['mean']), tensor_hash(data['scale'])])


def eligible(panels, arm, decoder):
    for seed in (47, 48):
        effects = [p[f'{arm}{seed}_{decoder}']['quality_effect'] for p in panels.values()]
        if any(e['lost_targets'] or any(c['fp'] > 0 for c in e['count_delta'].values()) for e in effects):
            return False
        if not any(c['fp'] < 0 for e in effects for c in e['count_delta'].values()):
            return False
    return True


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve shared quality trial')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    p, previous_audit, previous = checked_previous()
    _, parent_audit, _, _ = checked()
    extractors, classifiers, heads, training = {}, {}, {}, {}
    for seed in (47, 48):
        data = build(seed)
        _, extractors[seed], _ = load_parent(parent_audit, seed)
        for arm in p['arms']:
            head, training[f'{arm}{seed}'] = verify(arm, seed, data)
            heads[arm, seed] = (head, data['mean'], data['scale'])
        for key in ('initial_weights_sha256', 'features_sha256', 'normalization_sha256', 'optimizer_steps'):
            if training[f'iou{seed}'][key] != training[f'valid{seed}'][key]:
                raise ValueError('Matched trial differs beyond objective')
        record = parent_audit['classifiers'][str(seed)]
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Frozen classifier changed')
        classifiers[seed] = keras.models.load_model(record['path'], compile=False)
        print(dict(verified_seed=seed, rows=len(data['x']), mser_sources=len(data['mser_sources'])), flush=True)
    _, refs, _, _ = hybrid_inputs()
    incumbent = read_json(refs['joint']['raw']['path'])
    raw, panels = {}, {}
    for panel, sources in checked_panels().items():
        x = np.concatenate([prepare_input(rgb) for _, rgb in sources])
        outputs = {s: [v.numpy() for v in ex(x, training=False)] for s, ex in extractors.items()}
        frames = {f'{arm}{seed}_{decoder}': [] for arm in ('parent', *p['arms']) for seed in (47, 48) for decoder in DECODERS}
        for index, (frame, rgb) in enumerate(sources):
            for seed in (47, 48):
                values, maps = outputs[seed]
                old = previous[panel][f'parent{seed}_frozen']['frames'][index]
                if any(frame[k] != old[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                    raise ValueError('Frame identity changed')
                keys = ('box', 'crop_box', 'area', 'color_group', 'proposal_score')
                base = [{k: r[k] for k in keys} for r in old['detections'] if r['proposal_origin'] == 'mser']
                crops = classify(classifiers[seed], rgb, base)
                for decoder in DECODERS:
                    direct = as_direct(decode(values[index], decoder == 'independent_neighbor'))
                    rows = merge(crops, raw_direct(dict(detections=direct)), 'base_first')
                    parent = with_predictions(frame, rows)
                    parent.update(neural_calls=len(base), full_frame_neural_calls=1)
                    check_replayed_parent(parent, previous[panel][f'parent{seed}_{decoder}']['frames'][index])
                    frames[f'parent{seed}_{decoder}'].append(parent)
                    accepted = [r for r in rows if r['accepted']]
                    fx = np.stack([feature_row(maps[index], r) for r in accepted]) if accepted else None
                    for arm in p['arms']:
                        head, mean, scale = heads[arm, seed]
                        q = head((fx-mean)/scale, training=False).numpy()[:, 0].tolist() if accepted else []
                        it = iter(q)
                        predicted = gate(rows, [next(it) if r['accepted'] else None for r in rows])
                        result = with_predictions(frame, predicted)
                        result.update(neural_calls=len(base), full_frame_neural_calls=1,
                                      quality_head_calls=int(bool(accepted)), quality_rows=len(accepted))
                        frames[f'{arm}{seed}_{decoder}'].append(result)
        raw[panel] = {name: aggregate(fs) for name, fs in frames.items()}
        panels[panel] = {name: screen(result, incumbent[panel]['parent_parent47_current']) for name, result in raw[panel].items()}
        for seed in (47, 48):
            for decoder in DECODERS:
                for arm in p['arms']:
                    key = f'{arm}{seed}_{decoder}'
                    panels[panel][key]['quality_effect'] = difference(raw[panel][key], raw[panel][f'parent{seed}_{decoder}'])
                    panels[panel][key]['residual_origins'] = origins(raw[panel][key])
                panels[panel][f'valid{seed}_{decoder}']['objective_effect'] = difference(raw[panel][f'valid{seed}_{decoder}'], raw[panel][f'iou{seed}_{decoder}'])
        print(dict(panel=panel, valid={n: r['metrics'] for n, r in panels[panel].items() if n.startswith('valid')}), flush=True)
    gates = {f'{arm}_{decoder}': eligible(panels, arm, decoder) for arm in p['arms'] for decoder in DECODERS}
    write_json(RAW, raw)
    helpers = [__file__, 'scripts/shared_proposal_quality.py', 'scripts/summarize_proposal_quality_localizer.py',
               'scripts/audit_center_presence_localizer.py', 'scripts/audit_balanced_localizer.py',
               'scripts/diagnose_clutter_stage_attribution.py', 'scripts/audit_source_aware_balloon.py']
    write_json(OUTPUT, dict(inputs_sha256={**dependencies(), **{path: sha256(path) for path in helpers}},
        training=training, panels=panels, matched_head_gates=gates, raw=dict(path=str(RAW), sha256=sha256(RAW)),
        scope='Matched head-only objectives on frozen proposals/features.82TRAIN scenes, no heldout evaluation. Separate small head and up to24pooled candidate features add work; no latency claim.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(matched_head_gates=gates), flush=True)


if __name__ == '__main__':
    audit()
