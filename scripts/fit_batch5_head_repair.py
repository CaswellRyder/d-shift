"""Bounded deterministic head repair on sealed TRAIN data; no Pi mutation."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.batch5_occlusion_data import tensors, choose, new_scene_gate
from scripts.audit_batch5_occlusion import batch_scene_pixels
from scripts.batch5_head_constraints import assemble
from scripts.balloon_head_constraints import solve as feasible_solution
from scripts.balloon_minimum_drift import quadratic, solve
from scripts.fit_clutter_feature_correction import fold
from scripts.train_batch4_head_only import frozen_weights
from scripts.balloon_scene_constraints import RAW as TRAIN, RAW_SHA
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.train_batch3_retraining import calculate, correct

PROTOCOL = Path('configs/batch5-head-repair-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/batch5-head-repair-20261011.json')
RAW = Path('runs/batch5-head-repair-scenes-20261011.json')


def prepared(data, arm, parent, keras):
    rows = choose(data['batch_rows'][:-1], data['batch_rows'][-1], arm, data['protocol'])
    by_hash = {r['sha256']:x for r, x in zip(data['batch_rows'], data['batch_pixels'], strict=True)}
    chosen = np.stack([by_hash[r['sha256']] for r in rows])
    new_scene = batch_scene_pixels(data['batch_frames'])
    x = np.concatenate([data['x'][:2074], data['probes'], chosen, data['x'][-385:], new_scene])
    labels = np.r_[data['y'][:2074], data['probe_labels']]
    extractor = keras.Model(parent.inputs[0], parent.get_layer('spatial_features').output)
    features = extractor(x, training=False).numpy()
    if features.shape != (len(x), 128):
        raise ValueError('Frozen feature shape changed')
    mean = features[:2074].astype(float).mean(0)
    scale = np.maximum(features[:2074].astype(float).std(0), 1e-6)
    design = np.column_stack([(features.astype(float)-mean)/scale, np.ones(len(x))])
    logits = parent(x, training=False).numpy()
    batch_start = 2074+13+7+385
    hard = data['batch_rows'][-1]
    identities = [(f, d) for f in data['batch_frames'] for d in f['detections']]
    indices = [i for i, (f, d) in enumerate(identities)
               if f['source'] == hard['source_image'] and d['crop_box'] == hard['crop_box']]
    if len(indices) != 1:
        raise ValueError('Ambiguous hard proposal exemption')
    exempt = batch_start+indices[0]
    if not np.array_equal(x[exempt], data['batch_pixels'][-1]):
        raise ValueError('Hard constraint target differs from runtime proposal pixels')
    a, b, ids, objective = assemble(design, logits, labels, 2074, 7, 385, exempt)
    return dict(x=x, features=features, mean=mean, scale=scale, design=design, logits=logits,
        a=a, b=b, identities=ids, objective_indices=objective, h=quadratic(design[objective]),
        exempt=exempt, chosen_crop_sha256=[r['sha256'] for r in rows], batch_start=batch_start)


def fit():
    p = read_json(PROTOCOL)
    roots = {a:Path(f'runs/batch5-head-repair-{a}-20261011') for a in p['arms']}
    if OUTPUT.exists() or RAW.exists() or any(r.exists() for r in roots.values()):
        raise FileExistsError('Preserve head-repair experiment')
    data = tensors('control')
    if (sha256(p['parent']) != p['parent_sha256'] or p['parent'] != data['p']['parent']
            or sha256(TRAIN) != RAW_SHA):
        raise ValueError('Frozen parent or TRAIN geometry changed')
    original = read_json(TRAIN)
    if set(original) != {'training'}:
        raise ValueError('TRAIN-only scene boundary')
    old_frames = original['training']['parent']['frames']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    import scipy
    parent = keras.models.load_model(p['parent'], compile=False)
    if parent.count_params() != 7763:
        raise ValueError('Unexpected parent architecture')
    inputs = {a:prepared(data, a, parent, keras) for a in p['arms']}
    changed = np.flatnonzero(np.any(inputs['control']['x'] != inputs['hard']['x'], axis=(1, 2, 3))).tolist()
    if changed != [2092]:
        raise ValueError('Matched inputs differ beyond the selected same-photo crop')
    source_paths = [PROTOCOL, __file__, 'scripts/batch5_head_constraints.py', 'scripts/audit_batch5_occlusion.py',
        'scripts/balloon_head_constraints.py', 'scripts/balloon_minimum_drift.py',
        'scripts/balloon_false_acceptance_constraints.py', 'scripts/fit_clutter_feature_correction.py',
        'scripts/train_batch4_head_only.py', 'scripts/diagnose_clutter_round3_feature.py',
        'scripts/diagnose_clutter_stage_attribution.py', 'scripts/audit_balloon_context_distillation.py', TRAIN, p['parent']]
    sources = dict(data['sources_sha256'], **{str(f):sha256(f) for f in source_paths})
    v = inputs['control']
    old_parent = replay(old_frames, softmax(v['logits'][2094:2479]))
    new_parent = replay(data['batch_frames'], softmax(v['logits'][2479:]))
    for pred, expected in ((old_parent, old_frames), (new_parent, data['batch_frames'])):
        for a, b in zip(pred['frames'], expected, strict=True):
            check_replayed_parent(a, b)
    scenes = dict(parent=dict(original=old_parent, batch5=new_parent))
    results = {}
    for arm, root in roots.items():
        v = inputs[arm]
        root.mkdir(parents=True)
        write_json(root/'status.json', dict(state='solving_feasibility'))
        print(dict(arm=arm, stage='feasibility', constraints=len(v['b'])), flush=True)
        initial, feasibility = feasible_solution(v['a'], v['b'], time_limit=p['feasibility_seconds'])
        delta, solver = None, feasibility
        if initial is not None:
            write_json(root/'status.json', dict(state='minimizing_logit_drift'))
            print(dict(arm=arm, stage='minimum_drift', feasible=True), flush=True)
            try:
                delta, solver = solve(v['a'], v['b'], v['h'], initial, time_limit=p['quadratic_seconds'])
            except TimeoutError as exc:
                solver = dict(success=False, message=str(exc), timed_out=True)
        report = dict(feasibility=feasibility, solver=solver, constraints=len(v['b']),
            chosen_crop_sha256=v['chosen_crop_sha256'], exempt_proposal_index=v['exempt'],
            model_saved=False, training_gate_pass=False)
        archive = {k:v[k] for k in ('x', 'design', 'logits', 'a', 'b', 'identities', 'h',
                                    'mean', 'scale', 'objective_indices')}
        if initial is not None:
            archive['initial'] = initial
        if delta is not None:
            model = keras.models.load_model(p['parent'], compile=False)
            fold(model, delta.reshape(129, 2), v['mean'], v['scale'])
            frozen = all(np.array_equal(a, b) for a, b in zip(frozen_weights(parent), frozen_weights(model), strict=True))
            pk, pb = parent.get_layer('classifier').get_weights()
            mk, mb = model.get_layer('classifier').get_weights()
            if not frozen or not np.array_equal(pk[:, 0], mk[:, 0]) or pb[0] != mb[0]:
                raise ValueError('Frozen backbone/background changed')
            after = model(v['x'], training=False).numpy()
            expected = v['logits'].astype(float).copy()
            expected[:, 1:] += v['design']@delta.reshape(129, 2)
            maximum = float(np.max(np.abs(after-expected)))
            if not np.isfinite(after).all() or maximum > 1e-4:
                raise ValueError('Folded solution does not reproduce')
            old = replay(old_frames, softmax(after[2094:2479]))
            new = replay(data['batch_frames'], softmax(after[2479:]))
            effect = difference(old, old_parent)
            negatives = [false_acceptances(z[2074:2087]) for z in (v['logits'], after)]
            crop = calculate(v['logits'][:2074], after[:2074], data['y'][:2074],
                             after[2074:2087], data['probe_labels'])
            hard_logits = model(data['batch_pixels'][-1:], training=False).numpy()
            hard_ok = bool(correct(hard_logits, np.array([2]))[0])
            crop_retained = not crop['lost_old_rows'] and all(r['after'] <= r['before'] for r in crop['false_acceptances'].values())
            gate = bool(crop_retained and scene_gate(effect, *negatives) and new_scene_gate(new, hard_ok))
            selected_ok = correct(after[2087:2094], np.full(7, 2)).all()
            if not crop_retained or not selected_ok:
                raise ValueError('Folded crop bounds did not preserve required decisions')
            model.save(root/'student.keras')
            archive.update(delta=delta, after=after)
            report.update(model_saved=True, model_sha256=sha256(root/'student.keras'), fold_max_delta=maximum,
                frozen_weights_unchanged=True, original_metrics=old['metrics'], batch5_metrics=new['metrics'],
                effect=effect, background_false_acceptances=negatives, crop_diagnostics=crop,
                hard_blue_probability=float(softmax(hard_logits)[0, 2]), training_gate_pass=gate)
            scenes[arm] = dict(original=old, batch5=new)
        np.savez_compressed(root/'constraints.npz', **archive)
        report['archive_sha256'] = sha256(root/'constraints.npz')
        write_json(root/'report.json', report)
        write_json(root/'status.json', dict(state='complete'))
        results[arm] = dict(run=str(root), report_sha256=sha256(root/'report.json'), **report)
        print(dict(arm=arm, solved=solver['success'], gate=report['training_gate_pass'],
                   hard_blue=report.get('hard_blue_probability'), solver=solver), flush=True)
    if any(sha256(f) != h for f, h in sources.items()):
        raise ValueError('Sources changed during solving')
    write_json(RAW, scenes)
    write_json(OUTPUT, dict(results=results, sources_sha256=sources, protocol=p, scipy_version=scipy.__version__,
        changed_input_rows=changed, raw=dict(path=str(RAW), sha256=sha256(RAW)),
        eligible_for_independent_audit=results['hard']['training_gate_pass'],
        scope='Deterministic TRAIN-only constrained head fitting and scene recount. No runtime graph additions; speed unmeasured. Solver feasibility under sufficient bounds is not model-capacity proof. No DEV/export/deployment yet.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    fit()
