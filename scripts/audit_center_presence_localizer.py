"""Reload matched center-presence checkpoints and evaluate complete TRAIN scenes."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.center_presence_localizer_data import tensors, matched, dependencies, run_path, PROTOCOL
from scripts.public_scene_localizer_data import tensor_hash
from scripts.balloon_relative_box_loss import relative_loss
from scripts.balloon_peak_ranking_loss import local_peak_penalty, peak_ranking_loss
from scripts.audit_scale_balanced_search import checked_panels
from scripts.audit_direct_hybrid_balloon import checked_inputs as hybrid_inputs, raw_direct
from scripts.balloon_dense_runtime import prepare_input
from scripts.research_blue_peak_blend import decode
from scripts.audit_balloon_dense_proposals import supplement
from scripts.direct_localizer_balloon import select_direct
from scripts.independent_neighbor_decoder import decode as neighbor_decode
from scripts.balloon_center_presence_loss import center_presence_loss, center_presence_penalty
from scripts.direct_hybrid_balloon import merge
from scripts.research_confident_containment import select
from scripts.audit_source_aware_balloon import aggregate, with_predictions
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.diagnose_balanced_localizer_cells import inspect

OUTPUT = Path('research/balloon-red-blue-20261008/center-presence-localizer-20261011.json')
RAW = Path('runs/center-presence-localizer-scenes-20261011.json')
CELLS = Path('runs/center-presence-localizer-cells-20261011.json')
POLICIES = ('current', 'pooled', 'base_first', 'direct')
DECODERS = ('frozen', 'independent_neighbor')


def eligible_policies(panels):
    if set(panels) != {'original', 'window1', 'window2'}:
        raise ValueError('Incomplete TRAIN panels')
    gates = {}
    for decoder in DECODERS:
        for policy in POLICIES[:-1]:
            key = f'{decoder}_{policy}'
            gates[key] = all(
                all(p[f'presence{s}_{key}']['no_regression'] for p in panels.values()) and
                any(p[f'presence{s}_{key}']['effect']['gained_targets'] or
                    any(v['fp'] < 0 for v in p[f'presence{s}_{key}']['effect']['count_delta'].values())
                    for p in panels.values()) for s in (47, 48))
    return gates


def as_direct(proposals):
    found = []
    for proposal in proposals:
        group, probability = proposal['color_group'], proposal['proposal_score']
        scores = [1-probability, probability if group == 0 else 0., probability if group == 1 else 0.]
        found.append(dict(proposal, label=('red_balloon', 'blue_balloon')[group],
            score=probability, scores=scores, accepted=probability >= .5))
    return select_direct(found, .5)


def verify(arm, seed, d, parent):
    import keras
    import tensorflow as tf
    root = run_path(arm, seed)
    p, r = read_json(root/'provenance.json'), read_json(root/'report.json')
    if (read_json(root/'status.json')['state'] != 'complete' or p['arm'] != arm or p['seed'] != seed
            or p['sources_sha256'] != dependencies() or p['inputs_sha256'] != tensor_hash(d['x'])
            or p['targets_sha256'] != tensor_hash(d['y']) or p['fresh_optimizer'] is not True
            or p['fixed_final_checkpoint'] is not True or p['fit_frames'] != 193
            or p['excluded_original_holdout_frames'] != 16):
        raise ValueError('Training identity changed')
    for key in ('settings', 'parent', 'original_fit_indices', 'original_groups', 'negative_frames',
                'additional_frames', 'original_folds', 'appended_indices', 'base_settings', 'replay_settings'):
        if p[key] != d[key]:
            raise ValueError('Matched source metadata changed')
    if any(p[k] is not False for k in ('original_holdout_evaluated', 'development_evaluated',
            'test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved')):
        raise ValueError('Require TRAIN-only unpromoted trial')
    for key, filename in (('model_sha256', 'proposal.keras'), ('predictions_sha256', 'predictions.npy'),
                          ('provenance_sha256', 'provenance.json')):
        if sha256(root/filename) != r[key]:
            raise ValueError('Training artifacts changed')
    model = keras.models.load_model(root/'proposal.keras', custom_objects={
        'relative_loss': relative_loss, 'peak_ranking_loss': peak_ranking_loss, 'center_presence_loss': center_presence_loss})
    if (model.get_config() != parent.get_config() or model.count_params() != 2342
            or int(model.optimizer.iterations) != 5000 or r['optimizer_steps'] != 5000):
        raise ValueError('Graph or restored optimizer budget changed')
    with (root/'epochs.csv').open() as f:
        trace = list(csv.DictReader(f))
    if [int(t['epoch']) for t in trace] != list(range(200)) or not all(np.isfinite(float(t['loss'])) for t in trace):
        raise ValueError('Incomplete finite training trace')
    pred = model(d['x'], training=False).numpy()
    np.testing.assert_allclose(pred, np.load(root/'predictions.npy', allow_pickle=False), atol=1e-5, rtol=0)
    target, values = tf.constant(d['y']), tf.constant(pred)
    losses = dict(relative_fit_loss=float(relative_loss(target, values)), peak_penalty=float(local_peak_penalty(target, values)))
    losses['presence_penalty'] = float(center_presence_penalty(target, values))
    losses['fit_loss'] = losses['relative_fit_loss'] + losses['peak_penalty'] + (losses['presence_penalty'] if arm == 'presence' else 0)
    if any(not np.isclose(value, r[key], atol=1e-5, rtol=0) for key, value in losses.items()):
        raise ValueError('Saved losses do not reproduce')
    evidence = dict(run=str(root), parameters=2342, optimizer_steps=5000, **losses,
        artifacts_sha256={name: sha256(root/name) for name in
            ('proposal.keras', 'predictions.npy', 'provenance.json', 'report.json', 'epochs.csv')})
    return model, evidence


def classify(model, rgb, proposals):
    if not proposals:
        return []
    if len(proposals) > 12:
        raise ValueError('Crop-call budget exceeded')
    pixels = []
    for row in proposals:
        a, b, c, d = row['crop_box']
        if not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
            raise ValueError('Invalid inference crop')
        pixels.append(np.asarray(Image.fromarray(rgb[b:d, a:c]).resize((64, 64), Image.Resampling.BILINEAR), np.float32))
    probs = softmax(model(np.stack(pixels), training=False).numpy())
    found = []
    for row, prob in zip(proposals, probs, strict=True):
        k = int(prob.argmax())
        found.append(dict(row, label=('background', 'red_balloon', 'blue_balloon')[k],
            scores=prob.tolist(), score=float(prob[k]), accepted=bool(k > 0 and prob[k] >= .8)))
    return found


def audit():
    if OUTPUT.exists() or RAW.exists() or CELLS.exists():
        raise FileExistsError('Preserve center-presence trial')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    p = read_json(PROTOCOL)
    models, classifiers, training = {}, {}, {}
    if sha256(p['classifier_audit']) != p['classifier_audit_sha256']:
        raise ValueError('Frozen classifier audit changed')
    classifier_audit = read_json(p['classifier_audit'])
    if sha256(classifier_audit['raw']['path']) != classifier_audit['raw']['sha256']:
        raise ValueError('Frozen classifier replay changed')
    prior = read_json(p['parent_audit'])
    if sha256(p['parent_audit']) != p['parent_audit_sha256'] or sha256(prior['raw']['path']) != prior['raw']['sha256']:
        raise ValueError('Parent localizer replay changed')
    previous = read_json(prior['raw']['path'])
    for seed in (47, 48):
        arms = {a: tensors(a, seed) for a in ('control', 'presence')}
        matched(arms['control'], arms['presence'])
        parent = keras.models.load_model(p['parents'][str(seed)]['path'], compile=False)
        models[f'parent{seed}'] = parent
        for arm, d in arms.items():
            models[f'{arm}{seed}'], training[f'{arm}{seed}'] = verify(arm, seed, d, parent)
            print(dict(verified=f'{arm}{seed}', losses={k: v for k, v in training[f'{arm}{seed}'].items() if k.endswith('loss') or k == 'peak_penalty'}), flush=True)
        c = p['classifiers'][str(seed)]
        if sha256(c['path']) != c['sha256']:
            raise ValueError('Frozen classifier changed')
        record = classifier_audit['training'][f'corrected{seed}']
        for name, digest in record['artifacts_sha256'].items():
            if sha256(Path(record['run'])/name) != digest:
                raise ValueError('Frozen classifier artifact changed')
        for path, digest in read_json(Path(record['run'])/'provenance.json')['sources_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Frozen classifier source changed')
        classifiers[str(seed)] = keras.models.load_model(c['path'], compile=False)
    _, refs, _, _ = hybrid_inputs()
    incumbent = read_json(refs['joint']['raw']['path'])
    raw, panels, cells = {}, {}, {}
    for panel, sources in checked_panels().items():
        x = np.concatenate([prepare_input(rgb) for _, rgb in sources])
        predictions = {name: model(x, training=False).numpy() for name, model in models.items()}
        frames = {f'{name}_{decoder}_{policy}': [] for name in models for decoder in DECODERS for policy in POLICIES}
        cells[panel] = {name: [] for name in models}
        for index, (frame, rgb) in enumerate(sources):
            saved = previous[panel]['ranked47_base_first']['frames'][index]
            if any(frame[k] != saved[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                raise ValueError('Scene identity changed')
            keys = ('box', 'crop_box', 'area', 'color_group', 'proposal_score')
            base = [{k: r[k] for k in keys} for r in saved['detections'] if r['proposal_origin'] == 'mser']
            crops = {seed: classify(model, rgb, base) for seed, model in classifiers.items()}
            for name, outputs in predictions.items():
                seed, values = name[-2:], outputs[index]
                for decoder in DECODERS:
                    learned = decode(values, True) if decoder == 'frozen' else neighbor_decode(values)
                    current_proposals = supplement(base, learned)
                    direct = as_direct(learned)
                    candidates = {'current': select(classify(classifiers[seed], rgb, current_proposals)), 'direct': direct}
                    candidates.update({policy: merge(crops[seed], raw_direct(dict(detections=direct)), policy)
                                       for policy in ('pooled', 'base_first')})
                    for policy, selected in candidates.items():
                        row = with_predictions(frame, selected)
                        calls = 0 if policy == 'direct' else len(current_proposals) if policy == 'current' else len(base)
                        row.update(neural_calls=calls, full_frame_neural_calls=1)
                        if name.startswith('parent') and decoder == 'frozen':
                            check_replayed_parent(row, previous[panel][f'ranked{seed}_{policy}']['frames'][index])
                        frames[f'{name}_{decoder}_{policy}'].append(row)
                cells[panel][name].append(dict(source=frame['source'], targets=inspect(values, frame['truth'])))
        raw[panel] = {name: aggregate(fs) for name, fs in frames.items()}
        panels[panel] = {name: screen(result, incumbent[panel]['parent_parent47_current']) for name, result in raw[panel].items()}
        for name in models:
            for policy in POLICIES:
                key = f'{name}_independent_neighbor_{policy}'
                panels[panel][key]['decoder_effect'] = difference(raw[panel][key], raw[panel][f'{name}_frozen_{policy}'])
        for seed in (47, 48):
            for decoder in DECODERS:
                for policy in POLICIES:
                    name = f'presence{seed}_{decoder}_{policy}'
                    panels[panel][name]['matched_control_effect'] = difference(raw[panel][name], raw[panel][f'control{seed}_{decoder}_{policy}'])
                    panels[panel][name]['localizer_parent_effect'] = difference(raw[panel][name], raw[panel][f'parent{seed}_{decoder}_{policy}'])
        print(dict(panel=panel, candidates={n: v['metrics'] for n, v in panels[panel].items() if n.startswith('presence') and n.endswith('base_first')}), flush=True)

    gates = eligible_policies(panels)
    write_json(RAW, raw)
    write_json(CELLS, cells)
    helpers = [__file__, p['classifier_audit'], p['parent_audit'], 'scripts/independent_neighbor_decoder.py', 'scripts/balloon_dense_runtime.py', 'scripts/research_blue_peak_blend.py',
        'scripts/direct_localizer_balloon.py', 'scripts/direct_hybrid_balloon.py', 'scripts/research_confident_containment.py',
        'scripts/audit_balloon_dense_proposals.py', 'scripts/audit_direct_hybrid_balloon.py', 'scripts/audit_balanced_localizer.py',
        'scripts/diagnose_balanced_localizer_cells.py', 'scripts/diagnose_clutter_round3_feature.py',
        'scripts/diagnose_clutter_stage_attribution.py', 'scripts/audit_source_aware_balloon.py']
    counts = {panel: {name: dict(Counter(t['first_loss'] for f in fs for t in f['targets']))
                      for name, fs in groups.items()} for panel, groups in cells.items()}
    write_json(OUTPUT, dict(training=training, panels=panels, cells=dict(path=str(CELLS), sha256=sha256(CELLS)), cell_first_loss_counts=counts,
        eligible_for_separate_export_development=gates, raw=dict(path=str(RAW), sha256=sha256(RAW)),
        inputs_sha256={**dependencies(), **{str(path): sha256(path) for path in helpers}},
        classifiers=p['classifiers'],
        decision='Permit only separate export/DEV investigation for passing paired policy.' if any(gates.values()) else
                 'Reject center-presence/neighbor promotion: paired full-scene retention/improvement gate failed.',
        scope='Four verified fixed-final localizer retrainings;48variants on82TRAIN scenes. Both parent checkpoints reproduce prior direct-hybrid decisions on all panels. Identical inputs and2342parameters; training arms differ only in fitting objective; decoding is a separate fixed factor. Frozen perseed7763parameter classifiers. Cell diagnosis uses the frozen decoder and truth, not independent detection accuracy. No export, Pi timing or held-out qualification.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates), flush=True)


if __name__ == '__main__':
    audit()
