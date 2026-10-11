"""TRAIN-only minimum squared-logit drift, preserving old correct decisions explicitly.

This uses sufficient linear margin bounds, not an exhaustive feasibility test for
all possible softmax constraints. Solver failure is never deployment approval.
"""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_blue_balloon_batch4 import checked_extra, REVIEW
from scripts.balloon_head_constraints import constraints, solve as feasible_solution
from scripts.balloon_false_acceptance_constraints import constraints as rejection_constraints
from scripts.balloon_minimum_drift import quadratic, solve
from scripts.balloon_scene_constraints import scene_pixels, retention_constraints, decisions, RAW
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.grid_clutter_data import load_crop
from scripts.train_batch3_retraining import LABELS, choose_controls, correct, calculate
from scripts.train_batch4_head_only import frozen_weights

PROTOCOL = Path('configs/batch4-head-only-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/guarded-drift-balloon-head-20261011.json')


def fit():
    roots = {a: Path(f'runs/guarded-drift-balloon-head-{a}-20261011') for a in ('control', 'admit')}
    if OUTPUT.exists() or any(p.exists() for p in roots.values()):
        raise FileExistsError('Preserve existing constrained trial')
    p = read_json(PROTOCOL)
    rows, labels, old, _, _, previous = checked_inputs()
    extra = checked_extra()
    controls = choose_controls(rows, extra)
    probes = np.stack([load_crop(r)[0] for r in extra])
    new_labels = np.array([LABELS.index(r['label']) for r in extra])
    if len(old) != 2074 or len(extra) != 15 or sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Source or parent changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    import scipy
    parent = keras.models.load_model(p['parent'], compile=False)
    if parent.count_params() != 7763:
        raise ValueError('Architecture changed')
    extractor = keras.Model(parent.inputs[0], parent.get_layer('spatial_features').output)
    old_features = extractor(old, training=False).numpy()
    if old_features.shape != (2074, 128):
        raise ValueError('Feature shape changed')
    mean, scale = old_features.astype(float).mean(0), np.maximum(old_features.astype(float).std(0), 1e-6)
    before, probe_before = parent(old, training=False).numpy(), parent(probes, training=False).numpy()
    protected = correct(before, labels)
    scene, scene_identities = scene_pixels()
    scene_logits = parent(scene, training=False).numpy()
    scene_scores = np.asarray([r['scores'] for r in scene_identities])
    from dtr.runtime import softmax
    if not np.allclose(softmax(scene_logits), scene_scores, atol=1e-5, rtol=0):
        raise ValueError('Stored scene predictions differ from parent')
    scene_features = extractor(scene, training=False).numpy()
    scene_design = np.column_stack([(scene_features.astype(float)-mean)/scale, np.ones(len(scene))])
    scene_a, scene_b, scene_ids = retention_constraints(scene_design, scene_logits)
    paths = ['scripts/balloon_minimum_drift.py', 'scripts/balloon_false_acceptance_constraints.py', RAW, 'scripts/balloon_scene_constraints.py', 'scripts/prepare_blue_false_alarm_review.py', PROTOCOL, REVIEW, previous, __file__, 'scripts/balloon_head_constraints.py',
        'scripts/audit_blue_balloon_batch4.py', 'scripts/reviewed_balloon_batch.py',
        'scripts/train_batch3_retraining.py', 'scripts/train_batch4_head_only.py',
        'research/balloon-red-blue-20261008/blue-balloon-batch4-admission-20261011.json']
    sources = {str(s): sha256(s) for s in paths}
    results = {}
    for arm, root in roots.items():
        root.mkdir(parents=True)
        write_json(root/'status.json', dict(state='running'))
        chosen = extra if arm == 'admit' else controls
        additional = np.stack([load_crop(r)[0] for r in chosen])
        feature = np.concatenate([old_features, extractor(additional, training=False).numpy()])
        logits = np.concatenate([before, parent(additional, training=False).numpy()])
        y = np.r_[labels, [LABELS.index(r['label']) for r in chosen]]
        design = np.column_stack([(feature.astype(float)-mean)/scale, np.ones(len(feature))])
        a, b, identities = constraints(design, logits, y, np.r_[protected, np.zeros(15, bool)],
            np.r_[np.zeros(2074, bool), np.ones(15, bool)])
        identities += [(i+len(feature), t, r) for i, t, r in scene_ids]
        a, b = np.concatenate([a, scene_a]), np.r_[b, scene_b]
        reject_a, reject_b, reject_ids = rejection_constraints(design[:2074], before, labels)
        identities += reject_ids
        a, b = np.concatenate([a, reject_a]), np.r_[b, reject_b]
        initial, feasibility = feasible_solution(a, b, time_limit=60.)
        if initial is None:
            raise ValueError('Guarded TRAIN constraints infeasible: '+str(feasibility))
        objective_design = np.concatenate([design[:2074], scene_design])
        h = quadratic(objective_design)
        delta, solver = solve(a, b, h, initial, time_limit=120.)
        result = dict(feasibility=feasibility, new_wrong_color_constraints=len(reject_b), scene_rows=len(scene), scene_pixel_sha256=hashlib.sha256(scene.tobytes()).hexdigest(),
            scene_identities=scene_identities, solver=solver, constraints=len(b), protected_rows=int(protected.sum()),
            feature_sha256=hashlib.sha256(feature.tobytes()).hexdigest(),
            chosen_crop_sha256=[r['sha256'] for r in chosen],
            training_gate_pass=False, model_saved=False)
        stored = dict(a=a, b=b, identities=np.asarray(identities), mean=mean, scale=scale, h=h, initial=initial)
        if delta is not None:
            model = keras.models.load_model(p['parent'], compile=False)
            delta = delta.reshape(129, 2)
            kernel, bias = model.get_layer('classifier').get_weights()
            kernel[:, 1:] += (delta[:128]/scale[:, None]).astype(np.float32)
            bias[1:] += (delta[128]-np.sum(delta[:128]*mean[:, None]/scale[:, None], axis=0)).astype(np.float32)
            model.get_layer('classifier').set_weights([kernel, bias])
            if (not all(np.array_equal(v, w) for v, w in zip(frozen_weights(parent), frozen_weights(model), strict=True))
                    or not np.array_equal(kernel[:, 0], parent.get_layer('classifier').get_weights()[0][:, 0])
                    or bias[0] != parent.get_layer('classifier').get_weights()[1][0]):
                raise ValueError('Frozen backbone or background head changed')
            after, probe_after = model(old, training=False).numpy(), model(probes, training=False).numpy()
            actual = np.concatenate([after, model(additional, training=False).numpy()])
            expected = logits.astype(float).copy()
            expected[:, 1:] += design@delta
            difference = float(np.max(np.abs(actual-expected)))
            if not np.isfinite(actual).all() or difference > 1e-4:
                raise ValueError('Folded constraint solution differs')
            scene_after = model(scene, training=False).numpy()
            scene_retained = bool(np.array_equal(decisions(scene_logits), decisions(scene_after)))
            if not scene_retained:
                raise ValueError('Folded model lost TRAIN proposal decisions')
            result.update(calculate(before, after, labels, probe_after, new_labels),
                scene_decisions_retained=scene_retained,
                frozen_weights_unchanged=True, frozen_background_unchanged=True, fold_max_delta=difference,
                l1_head_change=float(np.abs(delta).sum()))
            model.save(root/'student.keras')
            np.savez_compressed(root/'predictions.npz', before=before, after=after, labels=labels,
                probe_before=probe_before, probe_after=probe_after, probe_labels=new_labels,
                scene_before=scene_logits, scene_after=scene_after)
            result.update(model_saved=True, model_sha256=sha256(root/'student.keras'),
                predictions_sha256=sha256(root/'predictions.npz'))
            stored['delta'] = delta
        np.savez_compressed(root/'constraints.npz', **stored)
        result['constraint_archive_sha256'] = sha256(root/'constraints.npz')
        write_json(root/'report.json', result)
        write_json(root/'status.json', dict(state='complete'))
        results[arm] = dict(run=str(root), report_sha256=sha256(root/'report.json'), **result)
        print(dict(arm=arm, solver=solver, gate=result['training_gate_pass'],
            lost=len(result.get('lost_old_rows', [])), accepted=sum(result.get('probe_correct', []))), flush=True)
    if any(sha256(s) != h for s, h in sources.items()):
        raise ValueError('Sources changed')
    write_json(OUTPUT, dict(results=results, sources_sha256=sources, scipy_version=scipy.__version__,
        parent_sha256=p['parent_sha256'], updated_parameters=258, total_parameters=7763,
        solver='SLSQP minimum mean squared TRAIN-logit delta plus0.001parameter ridge,120second cap per arm; convex KKT checks. Warm start from newly solved feasible L1 fit with wrong-color rejection bounds. Objective uses2074old views and385TRAIN proposals, excludes15new probes.',
        constraints='Old correct positive softmax0.8 with0.001margin safety; old background argmax margin0.001; selected new targets softmax>=0.81 sufficient pairwise bound. Runtime threshold remains0.8. Also retain385TRAIN proposal acceptance decisions with sufficient linear bounds; rejected proposals are not relabeled background. Additionally forbid new wrong-color acceptances on2074labeled TRAIN crops.',
        scope='Deterministic TRAIN-only feasibility and scene-decision constrained fitting, not seed-based SGD or independent accuracy. Linear bounds are sufficient, not necessary: infeasibility does not prove all frozen-feature approaches impossible.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    fit()



