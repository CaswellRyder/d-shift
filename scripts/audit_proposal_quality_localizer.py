"""Reproduce frozen parents and measure fixed quality gates on complete TRAIN panels."""
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.proposal_quality_data import PROTOCOL, build, dependencies, checked, load_parent, run_path
from scripts.proposal_quality_features import features, gate
from scripts.cached_balloon_decoder import decode
from scripts.audit_center_presence_localizer import classify, as_direct
from scripts.audit_direct_hybrid_balloon import raw_direct, checked_inputs as hybrid_inputs
from scripts.direct_hybrid_balloon import merge
from scripts.audit_scale_balanced_search import checked_panels
from scripts.balloon_dense_runtime import prepare_input
from scripts.audit_source_aware_balloon import with_predictions, aggregate
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.diagnose_clutter_round3_feature import check_replayed_parent

OUTPUT = Path('research/balloon-red-blue-20261008/proposal-quality-localizer-20261011.json')
RAW = Path('runs/proposal-quality-localizer-scenes-20261011.json')
DECODERS = ('frozen', 'independent_neighbor')


def verify(seed):
    import keras
    data, root = build(seed), run_path(seed)
    p, r = read_json(root/'provenance.json'), read_json(root/'report.json')
    if (read_json(root/'status.json')['state'] != 'complete' or p['seed'] != seed or
        p['sources_sha256'] != dependencies() or p['settings'] != data['settings'] or
        p['parent_sha256'] != data['parent_sha256'] or p['frozen_parent'] is not True or
        p['fixed_final_epoch'] is not True):
        raise ValueError('Training identity changed')
    for key, source in [('identities', 'identities'), ('source_records', 'records'),
        ('approved_complete_sources', 'approved_complete_sources'), ('excluded_incomplete_indices', 'excluded_incomplete_indices')]:
        if p[key] != data[source]:
            raise ValueError('TRAIN labels, source coverage or masks changed')
    for name, digest in r['artifacts_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Training artifact changed')
    with np.load(root/'training.npz', allow_pickle=False) as stored:
        for key in ('x', 'y', 'normalized', 'mean', 'scale', 'weights'):
            np.testing.assert_array_equal(stored[key], data[key])
    model = keras.models.load_model(root/'quality.keras')
    if model.count_params() != 265 or int(model.optimizer.iterations) != p['optimizer_steps'] or r['optimizer_steps'] != p['optimizer_steps']:
        raise ValueError('Head or restored schedule changed')
    with (root/'epochs.csv').open() as handle:
        trace = list(csv.DictReader(handle))
    if [int(t['epoch']) for t in trace] != list(range(200)) or not all(np.isfinite(float(t['loss'])) for t in trace):
        raise ValueError('Training trace incomplete')
    pred = model(data['normalized'], training=False).numpy()
    np.testing.assert_allclose(pred, np.load(root/'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0)
    mse = float(np.mean((pred[:, 0]-data['y'][:, 0])**2*data['weights']))
    if not np.isclose(mse, r['weighted_mse'], atol=1e-6, rtol=0):
        raise ValueError('Fitting loss does not reproduce')
    record = dict(run=str(root), **r, report_sha256=sha256(root/'report.json'),
                  approved_complete_sources=data['approved_complete_sources'], incomplete_rows_excluded=len(data['excluded_incomplete_indices']))
    return model, data['mean'], data['scale'], record


def eligible(panels, decoder):
    for seed in (47, 48):
        key = f'quality{seed}_{decoder}'
        rows = [panel[key]['quality_effect'] for panel in panels.values()]
        if any(r['lost_targets'] or any(c['fp'] > 0 for c in r['count_delta'].values()) for r in rows):
            return False
        if not any(c['fp'] < 0 for r in rows for c in r['count_delta'].values()):
            return False
    return True


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve quality experiment')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    p, parent_audit, _, _ = checked()
    if sha256(parent_audit['raw']['path']) != parent_audit['raw']['sha256']:
        raise ValueError('Frozen parent replay changed')
    previous = read_json(parent_audit['raw']['path'])
    extractors, heads, classifiers, training = {}, {}, {}, {}
    for seed in (47, 48):
        _, extractors[seed], _ = load_parent(parent_audit, seed)
        head, mean, scale, training[str(seed)] = verify(seed)
        heads[seed] = (head, mean, scale)
        record = parent_audit['classifiers'][str(seed)]
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Frozen classifier changed')
        classifiers[seed] = keras.models.load_model(record['path'], compile=False)
        print(dict(verified=seed, training_proposals=training[str(seed)]['fit_proposals']), flush=True)
    _, refs, _, _ = hybrid_inputs()
    incumbent = read_json(refs['joint']['raw']['path'])
    raw, panels = {}, {}
    for panel, sources in checked_panels().items():
        x = np.concatenate([prepare_input(rgb) for _, rgb in sources])
        outputs = {seed: [v.numpy() for v in extractor(x, training=False)] for seed, extractor in extractors.items()}
        frames = {f'{arm}{seed}_{decoder}': [] for arm in ('parent', 'quality') for seed in (47, 48) for decoder in DECODERS}
        for index, (frame, rgb) in enumerate(sources):
            for seed in (47, 48):
                values, maps = outputs[seed]
                old = previous[panel][f'presence{seed}_frozen_base_first']['frames'][index]
                if any(frame[k] != old[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Source identity changed')
                keys = ('box', 'crop_box', 'area', 'color_group', 'proposal_score')
                base = [{k: row[k] for k in keys} for row in old['detections'] if row['proposal_origin'] == 'mser']
                crops = classify(classifiers[seed], rgb, base)
                for decoder in DECODERS:
                    proposals = decode(values[index], decoder == 'independent_neighbor')
                    direct = as_direct(proposals)
                    parent_rows = merge(crops, raw_direct(dict(detections=direct)), 'base_first')
                    parent_frame = with_predictions(frame, parent_rows)
                    parent_frame.update(neural_calls=len(base), full_frame_neural_calls=1)
                    saved = previous[panel][f'presence{seed}_{decoder}_base_first']['frames'][index]
                    check_replayed_parent(parent_frame, saved)
                    learned = [r for r in parent_rows if r['proposal_origin'] == 'learned']
                    qvalues = []
                    if learned:
                        head, mean, scale = heads[seed]
                        fx = np.stack([features(maps[index], row) for row in learned])
                        qvalues = head((fx-mean)/scale, training=False).numpy()[:, 0].tolist()
                    it = iter(qvalues)
                    quality_rows = gate(parent_rows, [next(it) if r['proposal_origin'] == 'learned' else None for r in parent_rows])
                    quality_frame = with_predictions(frame, quality_rows)
                    quality_frame.update(neural_calls=len(base), full_frame_neural_calls=1,
                                         quality_head_calls=int(bool(learned)), quality_rows=len(learned))
                    frames[f'parent{seed}_{decoder}'].append(parent_frame)
                    frames[f'quality{seed}_{decoder}'].append(quality_frame)
        raw[panel] = {name: aggregate(rows) for name, rows in frames.items()}
        panels[panel] = {name: screen(result, incumbent[panel]['parent_parent47_current']) for name, result in raw[panel].items()}
        for seed in (47, 48):
            for decoder in DECODERS:
                key = f'quality{seed}_{decoder}'
                panels[panel][key]['quality_effect'] = difference(raw[panel][key], raw[panel][f'parent{seed}_{decoder}'])
        print(dict(panel=panel, quality={name: row['metrics'] for name, row in panels[panel].items() if name.startswith('quality')}), flush=True)
    gates = {decoder: eligible(panels, decoder) for decoder in DECODERS}
    write_json(RAW, raw)
    helpers = [__file__, PROTOCOL, 'scripts/proposal_quality_features.py', 'scripts/cached_balloon_decoder.py',
        'scripts/audit_center_presence_localizer.py', 'scripts/direct_hybrid_balloon.py', 'scripts/audit_source_aware_balloon.py',
        'scripts/audit_balanced_localizer.py', 'scripts/diagnose_clutter_stage_attribution.py', 'scripts/diagnose_clutter_round3_feature.py']
    write_json(OUTPUT, dict(inputs_sha256={**dependencies(), **{str(path): sha256(path) for path in helpers}},
        training=training, panels=panels, matched_head_gates=gates,
        raw=dict(path=str(RAW), sha256=sha256(RAW)),
        scope='Fixed final head checkpoints; head-only quality gating versus unchanged parent on82TRAIN scenes. Fresh frozen localizer/classifier inference and exact parent-decision replay checks. Added small head calls are separate from crop/localizer counts. No latency or qualification claim.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(matched_head_gates=gates), flush=True)


if __name__ == '__main__':
    audit()
