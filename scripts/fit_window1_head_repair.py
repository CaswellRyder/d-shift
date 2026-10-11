"""Bounded fixed-feature repair of priority classifiers using TRAIN evidence only."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.window1_priority_data import tensors
from scripts.window1_trial_data import recovered_targets
from scripts.audit_window1_runtime import batch_scene_pixels
from scripts.audit_window1_priority import OUTPUT as PRIORITY
from scripts.window1_repair_constraints import assemble
from scripts.balloon_head_constraints import solve as feasible_solution
from scripts.balloon_minimum_drift import quadratic, solve
from scripts.fit_clutter_feature_correction import fold
from scripts.train_batch4_head_only import frozen_weights
from scripts.balloon_scene_constraints import RAW as TRAIN, RAW_SHA
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.train_batch3_retraining import calculate, correct

PROTOCOL = Path('configs/window1-head-repair-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/window1-head-repair-20261011.json')
RAW = Path('runs/window1-head-repair-scenes-20261011.json')
OLD_START, NEW_START = 2096, 2481


def prepared(data, parent, model, keras):
    chosen = data['x'][2126:2135]
    chosen_labels = data['y'][2126:2135]
    x = np.concatenate([data['x'][:2074], data['probes'], chosen, data['x'][-385:],
                        batch_scene_pixels(data['batch_frames'])])
    reference = parent(x, training=False).numpy()
    current = model(x, training=False).numpy()
    features = keras.Model(model.inputs[0], model.get_layer('spatial_features').output)(x, training=False).numpy()
    if features.shape != (len(x), 128):
        raise ValueError('Feature dimensions changed')
    mean, scale = features[:2074].astype(float).mean(0), np.maximum(features[:2074].astype(float).std(0), 1e-6)
    design = np.column_stack([(features.astype(float)-mean)/scale, np.ones(len(x))])
    ids = [(f, d) for f in data['batch_frames'] for d in f['detections']]
    exempt = []
    for row, pixels in zip(data['batch_rows'][-2:], data['batch_pixels'][-2:], strict=True):
        matches = [i for i, (f, d) in enumerate(ids)
                   if f['source'] == row['source_image'] and d['crop_box'] == row['crop_box']]
        if len(matches) != 1 or not np.array_equal(x[NEW_START+matches[0]], pixels):
            raise ValueError('Ambiguous or altered reviewed runtime exemption')
        exempt.append(NEW_START+matches[0])
    labels = np.r_[data['y'][:2074], data['probe_labels']]
    a, b, identities, objective = assemble(design, reference, current, labels, chosen_labels, 385, exempt)
    return dict(x=x, reference=reference, current=current, design=design, mean=mean, scale=scale,
                a=a, b=b, identities=identities, objective=objective, h=quadratic(design[objective]),
                exempt=np.array(exempt), chosen_labels=chosen_labels)


def fit():
    p, priority = read_json(PROTOCOL), read_json(PRIORITY)
    roots = {s:Path(f'runs/window1-head-repair-seed{s}-20261011') for s in p['seeds']}
    if OUTPUT.exists() or RAW.exists() or any(r.exists() for r in roots.values()):
        raise FileExistsError('Preserve bounded repair attempt')
    data = tensors('priority')
    if sha256(TRAIN) != RAW_SHA or any(sha256(f) != h for f, h in priority['inputs_sha256'].items()):
        raise ValueError('Frozen TRAIN experiment changed')
    old = read_json(TRAIN)
    if set(old) != {'training'}:
        raise ValueError('TRAIN-only geometry')
    old_frames = old['training']['parent']['frames']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    import scipy
    parent = keras.models.load_model(data['p']['parent'], compile=False)
    sources = dict(data['sources_sha256'])
    sources.update({str(f):sha256(f) for f in (PROTOCOL, PRIORITY, TRAIN, __file__,
        'scripts/window1_repair_constraints.py', 'scripts/audit_window1_runtime.py',
        'scripts/balloon_head_constraints.py', 'scripts/balloon_false_acceptance_constraints.py',
        'scripts/balloon_minimum_drift.py', 'scripts/fit_clutter_feature_correction.py',
        'scripts/train_batch4_head_only.py', 'scripts/diagnose_clutter_round3_feature.py',
        'scripts/diagnose_clutter_stage_attribution.py', 'scripts/audit_balloon_context_distillation.py')})
    results, scenes = {}, {}
    for seed, root in roots.items():
        prior = priority['results'][f'priority{seed}']
        prior_root = Path(prior['run'])
        if any(sha256(prior_root/f) != h for f, h in prior['artifacts_sha256'].items()):
            raise ValueError('Priority model artifacts changed')
        path = prior_root/'student.keras'
        sources[str(path)] = sha256(path)
        model = keras.models.load_model(path, compile=False)
        if model.count_params() != p['parameters'] or model.get_config() != parent.get_config():
            raise ValueError('Architecture changed')
        v = prepared(data, parent, model, keras)
        original = replay(old_frames, softmax(v['reference'][OLD_START:NEW_START]))
        new_original = replay(data['batch_frames'], softmax(v['reference'][NEW_START:]))
        for result, expected in ((original, old_frames), (new_original, data['batch_frames'])):
            for a, b in zip(result['frames'], expected, strict=True):
                check_replayed_parent(a, b)
        scenes['parent'] = dict(original=original, window1=new_original)
        root.mkdir(parents=True)
        write_json(root/'status.json', dict(state='solving_feasibility'))
        print(dict(seed=seed, stage='feasibility', constraints=len(v['b'])), flush=True)
        initial, feasibility = feasible_solution(v['a'], v['b'], time_limit=p['feasibility_seconds'])
        delta, solver = None, feasibility
        if initial is not None:
            write_json(root/'status.json', dict(state='minimum_drift'))
            print(dict(seed=seed, stage='minimum_drift'), flush=True)
            try:
                delta, solver = solve(v['a'], v['b'], v['h'], initial, time_limit=p['quadratic_seconds'])
            except TimeoutError as exc:
                solver = dict(success=False, timed_out=True, message=str(exc))
        report = dict(seed=seed, source_model=str(path), source_sha256=sha256(path),
            feasibility=feasibility, solver=solver, constraints=len(v['b']),
            model_saved=False, training_gate_pass=False)
        archive = dict(v)
        if initial is not None:
            archive['initial'] = initial
        if delta is not None:
            frozen = frozen_weights(model)
            pk, pb = model.get_layer('classifier').get_weights()
            fold(model, delta.reshape(129, 2), v['mean'], v['scale'])
            mk, mb = model.get_layer('classifier').get_weights()
            if (not all(np.array_equal(a, b) for a, b in zip(frozen, frozen_weights(model), strict=True))
                    or not np.array_equal(pk[:, 0], mk[:, 0]) or pb[0] != mb[0]):
                raise ValueError('Frozen representation/background changed')
            after = model(v['x'], training=False).numpy()
            expected = v['current'].astype(float).copy()
            expected[:, 1:] += v['design']@delta.reshape(129, 2)
            error = float(np.max(np.abs(after-expected)))
            if not np.isfinite(after).all() or error > 1e-4:
                raise ValueError('Folded logits differ from constrained solution')
            old_after = replay(old_frames, softmax(after[OLD_START:NEW_START]))
            new_after = replay(data['batch_frames'], softmax(after[NEW_START:]))
            effect, new_effect = difference(old_after, original), difference(new_after, new_original)
            negatives = [false_acceptances(z[2074:2087]) for z in (v['reference'], after)]
            crop = calculate(v['reference'][:2074], after[:2074], data['y'][:2074], after[2074:2087], data['probe_labels'])
            crop_retained = not crop['lost_old_rows'] and all(r['after'] <= r['before'] for r in crop['false_acceptances'].values())
            chosen_ok = bool(correct(after[2087:2096], v['chosen_labels']).all())
            if not crop_retained or not chosen_ok:
                raise ValueError('Folded solution violates supervised bounds')
            gate = bool(scene_gate(effect, *negatives) and scene_gate(new_effect, *negatives)
                        and recovered_targets(new_after) and crop_retained and chosen_ok)
            model.save(root/'student.keras')
            archive.update(delta=delta, after=after)
            report.update(model_saved=True, model_sha256=sha256(root/'student.keras'),
                fold_max_delta=error, frozen_weights_unchanged=True, original_metrics=old_after['metrics'],
                window1_metrics=new_after['metrics'], effect=effect, new_effect=new_effect,
                crop_diagnostics=crop, background_false_acceptances=negatives,
                runtime_blue_probabilities=softmax(after[v['exempt']])[:, 2].tolist(), training_gate_pass=gate)
            scenes[str(seed)] = dict(original=old_after, window1=new_after)
        np.savez_compressed(root/'constraints.npz', **archive)
        report['archive_sha256'] = sha256(root/'constraints.npz')
        write_json(root/'report.json', report)
        write_json(root/'status.json', dict(state='complete'))
        results[str(seed)] = dict(run=str(root), report_sha256=sha256(root/'report.json'), **report)
        print(dict(seed=seed, solved=solver['success'], gate=report['training_gate_pass'],
                   metrics=report.get('window1_metrics'), solver=solver), flush=True)
    if any(sha256(f) != h for f, h in sources.items()):
        raise ValueError('Repair inputs changed')
    write_json(RAW, scenes)
    write_json(OUTPUT, dict(results=results, sources_sha256=sources, protocol=p, scipy_version=scipy.__version__,
        raw=dict(path=str(RAW), sha256=sha256(RAW)), eligible_for_independent_audit=all(r['training_gate_pass'] for r in results.values()),
        scope='TRAIN constrained head repair and replay only. Same inference graph; actual speed unmeasured. Independent solver/model reconstruction and export/full-search validation still required.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    fit()
