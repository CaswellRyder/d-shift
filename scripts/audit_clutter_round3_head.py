"""Reconstruct constrained TRAIN inputs, bounds and folded checkpoint predictions."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_object_round3_data import checked_extra
from scripts.balloon_head_constraints import constraints, solve as feasible_solution
from scripts.balloon_minimum_drift import quadratic, certificate
from scripts.balloon_false_acceptance_constraints import constraints as rejection_constraints
from scripts.balloon_scene_constraints import scene_pixels, retention_constraints, decisions
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.fit_clutter_round3_head import OUTPUT as FIT, PROTOCOL
from scripts.grid_clutter_data import load_crop
from scripts.train_batch3_retraining import LABELS, choose_controls, correct, calculate
from scripts.train_batch4_head_only import frozen_weights

OUTPUT = Path('research/balloon-red-blue-20261008/clutter-round3-head-audit-20261011.json')


def checked_fit():
    fit = read_json(FIT)
    if (set(fit['results']) != {'control', 'admit'}
            or fit['updated_parameters'] != 258 or fit['total_parameters'] != 7763
            or fit['parent_sha256'] != read_json(PROTOCOL)['parent_sha256']
            or any(fit[k] is not False for k in ('development_evaluated', 'test_evaluated',
                                                'pi_tested', 'deployment_approved', 'qualification_achieved'))
            or any(sha256(p) != h for p, h in fit['sources_sha256'].items())):
        raise ValueError('Constrained fitting provenance changed')
    for row in fit['results'].values():
        root = Path(row['run'])
        report = read_json(root/'report.json')
        if (read_json(root/'status.json')['state'] != 'complete'
                or sha256(root/'report.json') != row['report_sha256']
                or any(report[k] != row[k] for k in report)):
            raise ValueError('Incomplete or changed constrained checkpoint')
        for field, file in (('model_sha256', 'student.keras'), ('predictions_sha256', 'predictions.npz'),
                            ('constraint_archive_sha256', 'constraints.npz')):
            if field != 'constraint_archive_sha256' and not row['model_saved']:
                continue
            if sha256(root/file) != row[field]:
                raise ValueError('Constrained artifact identity changed')
    return fit


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    fit = checked_fit()
    p = read_json(PROTOCOL)
    rows, labels, old, _, _, _ = checked_inputs()
    extra = checked_extra()
    probes = np.stack([load_crop(r)[0] for r in extra])
    probe_labels = np.array([LABELS.index(r['label']) for r in extra])
    if len(old) != 2074 or len(extra) != 13 or sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent or TRAIN inputs changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'], compile=False)
    extractor = keras.Model(parent.inputs[0], parent.get_layer('spatial_features').output)
    old_features = extractor(old, training=False).numpy()
    mean, scale = old_features.astype(float).mean(0), np.maximum(old_features.astype(float).std(0), 1e-6)
    before, probe_before = parent(old, training=False).numpy(), parent(probes, training=False).numpy()
    protected = correct(before, labels)
    scene, scene_identities = scene_pixels()
    scene_logits = parent(scene, training=False).numpy()
    scene_features = extractor(scene, training=False).numpy()
    scene_design = np.column_stack([(scene_features.astype(float)-mean)/scale, np.ones(len(scene))])
    scene_a, scene_b, scene_ids = retention_constraints(scene_design, scene_logits)
    np.testing.assert_allclose(softmax(scene_logits), [r['scores'] for r in scene_identities], atol=1e-5, rtol=0)
    results = {}
    for arm, row in fit['results'].items():
        root = Path(row['run'])
        chosen = extra if arm == 'admit' else choose_controls(rows, extra)
        additional = np.stack([load_crop(r)[0] for r in chosen])
        feature = np.concatenate([old_features, extractor(additional, training=False).numpy()])
        logits = np.concatenate([before, parent(additional, training=False).numpy()])
        y = np.r_[labels, [LABELS.index(r['label']) for r in chosen]]
        design = np.column_stack([(feature.astype(float)-mean)/scale, np.ones(len(feature))])
        a, b, ids = constraints(design, logits, y, np.r_[protected, np.zeros(13, bool)],
                               np.r_[np.zeros(2074, bool), np.ones(13, bool)])
        ids += [(i+len(feature), t, r) for i, t, r in scene_ids]
        a, b = np.concatenate([a, scene_a]), np.r_[b, scene_b]
        reject_a, reject_b, reject_ids = rejection_constraints(design[:2074], before, labels)
        ids += reject_ids
        a, b = np.concatenate([a, reject_a]), np.r_[b, reject_b]
        if row['new_wrong_color_constraints'] != len(reject_b):
            raise ValueError('Wrong-color constraint count changed')
        if (row['scene_rows'] != len(scene) or row['scene_identities'] != scene_identities
                or row['scene_pixel_sha256'] != hashlib.sha256(scene.tobytes()).hexdigest()):
            raise ValueError('Scene constraint inputs differ')
        if (hashlib.sha256(feature.tobytes()).hexdigest() != row['feature_sha256']
                or row['chosen_crop_sha256'] != [r['sha256'] for r in chosen]
                or row['protected_rows'] != int(protected.sum()) or row['constraints'] != len(b)):
            raise ValueError('Constraint inputs differ')
        h = quadratic(np.concatenate([design[:2074], scene_design]))
        with np.load(root/'constraints.npz', allow_pickle=False) as stored:
            for key, value in dict(a=a, b=b, identities=np.asarray(ids), mean=mean, scale=scale, h=h).items():
                np.testing.assert_allclose(stored[key], value, atol=1e-10, rtol=0)
            if not row['model_saved']:
                if row['training_gate_pass'] or 'delta' in stored.files:
                    raise ValueError('Unsupported failed-fit claim')
                candidate, feasibility = feasible_solution(a, b, time_limit=60.)
                if candidate is not None or feasibility['status'] != 2 or row['feasibility']['status'] != 2:
                    raise ValueError('Infeasibility not independently reproduced')
                results[arm] = row
                print(dict(audited=arm, infeasible=True, model_saved=False), flush=True)
                continue
            initial, delta = stored['initial'], stored['delta']
            if (row['feasibility']['success'] is not True or np.max(b-a@initial, initial=0) > 1e-6
                    or abs(np.abs(initial).sum()-row['feasibility']['objective']) > 1e-8):
                raise ValueError('Feasible initialization changed')
        if delta.shape != (129, 2) or not np.isfinite(delta).all():
            raise ValueError('Invalid head update')
        violation = float(np.max(b-a@delta.ravel(), initial=0))
        proof = certificate(h, a, b, delta.ravel())
        if (any(abs(proof[k]-row['solver'][k]) > 1e-8 for k in proof)
                or proof['stationarity'] > 1e-4 or proof['complementarity'] > 1e-4
                or abs(.5*initial@h@initial-row['solver']['initial_objective']) > 1e-8):
            raise ValueError('Minimum-drift optimality check failed')
        if (violation > 1e-6 or abs(np.abs(delta).sum()-row['l1_head_change']) > 1e-8
                or abs(violation-row['solver']['maximum_violation']) > 1e-8):
            raise ValueError('Constraint solution invalid')
        model = keras.models.load_model(root/'student.keras', compile=False)
        if model.count_params() != 7763 or not all(np.array_equal(v, w) for v, w in
                zip(frozen_weights(parent), frozen_weights(model), strict=True)):
            raise ValueError('Backbone changed')
        kernel, bias = parent.get_layer('classifier').get_weights()
        kernel[:, 1:] += (delta[:128]/scale[:, None]).astype(np.float32)
        bias[1:] += (delta[128]-np.sum(delta[:128]*mean[:, None]/scale[:, None], axis=0)).astype(np.float32)
        for expected, actual in zip((kernel, bias), model.get_layer('classifier').get_weights(), strict=True):
            np.testing.assert_array_equal(actual, expected)
        after, probe_after = model(old, training=False).numpy(), model(probes, training=False).numpy()
        chosen_after = model(additional, training=False).numpy()
        expected = logits.astype(float).copy()
        expected[:, 1:] += design@delta
        fold_delta = float(np.max(np.abs(np.concatenate([after, chosen_after])-expected)))
        if (fold_delta > 1e-4 or abs(fold_delta-row['fold_max_delta']) > 1e-8
                or not (chosen_after[:, 0, None]-chosen_after[:, 1:] >= .001-5e-5).all()):
            raise ValueError('Folded bounds fail')
        scene_after = model(scene, training=False).numpy()
        if row['scene_decisions_retained'] is not True or not np.array_equal(decisions(scene_logits), decisions(scene_after)):
            raise ValueError('TRAIN scene proposal decisions changed')
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            for key, value in dict(before=before, after=after, probe_before=probe_before,
                                   probe_after=probe_after, labels=labels, probe_labels=probe_labels,
                                   scene_before=scene_logits, scene_after=scene_after).items():
                np.testing.assert_allclose(stored[key], value, atol=1e-5, rtol=0)
        if arm == 'control' and np.max(b, initial=0) == 0 and (np.any(delta) or not np.array_equal(before, after)):
            raise ValueError('Feasible zero-change control did not remain unchanged')
        measured = calculate(before, after, labels, probe_after, probe_labels)
        if any(row[k] != v for k, v in measured.items()):
            raise ValueError('TRAIN gate does not reproduce')
        results[arm] = row
        print(dict(audited=arm, gate=measured['training_gate_pass'],
                   new_accepted=sum(measured['probe_correct']), lost=len(measured['lost_old_rows'])), flush=True)
    checked_fit()
    write_json(OUTPUT, dict(results=results, fit_sha256=sha256(FIT), script_sha256=sha256(__file__),
        sources_sha256={p: sha256(p) for p in ('scripts/fit_blue_prototype_rejection.py',
            'scripts/grid_clutter_data.py', 'scripts/train_batch3_retraining.py',
            'scripts/balloon_head_constraints.py', 'scripts/balloon_scene_constraints.py', 'scripts/balloon_minimum_drift.py', 'scripts/balloon_false_acceptance_constraints.py',
            'scripts/blue_object_round3_data.py', 'research/balloon-red-blue-20261008/blue-object-round3-admission-20261011.json')},
        eligible_for_full_scene_comparison=results['admit']['training_gate_pass'],
        scope='Deterministic constrained TRAIN fit independently reconstructed: frozen backbone, convex objective and KKT residuals, all crop and scene bounds, folded head and checkpoint predictions,385TRAIN proposal decision parity. Both arms use identical constraints and objectives;13novel background crops versus13existing backgrounds. Any infeasibility refers only to these sufficient linear bounds. No scene or independent accuracy claim.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()





