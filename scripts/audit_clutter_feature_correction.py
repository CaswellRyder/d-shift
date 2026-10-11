"""Reconstruct corrected-feature constraints, certificates and saved predictions."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.clutter_round3_feature_data import tensors, hashes
from scripts.fit_clutter_feature_correction import PROTOCOL, OUTPUT as FIT, checked_sources, prepared
from scripts.balloon_minimum_drift import certificate
from scripts.balloon_head_constraints import solve as feasible_solution
from scripts.balloon_scene_constraints import decisions
from scripts.train_batch3_retraining import calculate
from scripts.train_batch4_head_only import frozen_weights

OUTPUT = Path('research/balloon-red-blue-20261008/clutter-feature-correction-audit-20261011.json')


def checked_fit():
    p, feature_audit = checked_sources()
    fit = read_json(FIT)
    if (set(fit['results']) != set(p['names']) or fit['updated_parameters'] != 258
            or fit['total_parameters'] != 7763
            or any(sha256(k) != h for k, h in fit['sources_sha256'].items())
            or any(fit[k] is not False for k in ('development_evaluated', 'test_evaluated', 'pi_tested',
                                                'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Correction provenance changed')
    for name, row in fit['results'].items():
        root = Path(row['run'])
        expected = Path(feature_audit['results'][name]['run'])/'student.keras'
        report = read_json(root/'report.json')
        if (read_json(root/'status.json')['state'] != 'complete'
                or row['feature_path'] != str(expected) or row['feature_sha256'] != sha256(expected)
                or row['report_sha256'] != sha256(root/'report.json')
                or any(report[k] != row[k] for k in report)):
            raise ValueError('Correction checkpoint changed')
        for key, file in (('constraint_archive_sha256','constraints.npz'), ('model_sha256','student.keras'),
                          ('predictions_sha256','predictions.npz')):
            if key != 'constraint_archive_sha256' and not row['model_saved']:
                continue
            if row[key] != sha256(root/file):
                raise ValueError('Correction artifact changed')
    return p, fit


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p, fit = checked_fit()
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    reference = keras.models.load_model(p['reference'], compile=False)
    cache = {a: tensors(a) for a in ('control', 'admit')}
    for name, row in fit['results'].items():
        data = cache['admit' if name.startswith('admit') else 'control']
        root = Path(row['run'])
        feature = keras.models.load_model(row['feature_path'], compile=False)
        v = prepared(data, reference, feature, keras)
        if row['input_sha256'] != hashes(data) or row['constraints'] != len(v['b']):
            raise ValueError('Reconstructed data changed')
        with np.load(root/'constraints.npz', allow_pickle=False) as stored:
            for k in ('a', 'b', 'identities', 'h', 'mean', 'scale'):
                np.testing.assert_allclose(stored[k], v[k], atol=1e-10, rtol=0)
            if not row['model_saved']:
                candidate, result = feasible_solution(v['a'], v['b'], time_limit=60.)
                if (candidate is not None or result['status'] != 2 or row['feasibility']['status'] != 2
                        or row['training_gate_pass'] or 'delta' in stored.files):
                    raise ValueError('Infeasibility not independently reproduced')
                print(dict(audited=name, infeasible=True), flush=True)
                continue
            initial, delta = stored['initial'], stored['delta']
        if delta.shape != (129, 2) or not np.isfinite(delta).all():
            raise ValueError('Invalid correction')
        proof = certificate(v['h'], v['a'], v['b'], delta.ravel())
        if (np.max(v['b']-v['a']@initial, initial=0)>1e-6
                or abs(np.abs(initial).sum()-row['feasibility']['objective'])>1e-8
                or abs(.5*initial@v['h']@initial-row['solver']['initial_objective'])>1e-8
                or any(abs(value-row['solver'][k])>1e-8 for k,value in proof.items())
                or proof['maximum_violation']>1e-6 or proof['stationarity']>1e-4 or proof['complementarity']>1e-4
                or row['solver']['success'] is not True):
            raise ValueError('Constraint/KKT proof failed')
        model = keras.models.load_model(root/'student.keras', compile=False)
        if (model.count_params()!=7763 or model.get_config()!=feature.get_config()
                or not all(np.array_equal(a,b) for a,b in zip(frozen_weights(feature), frozen_weights(model), strict=True))):
            raise ValueError('Frozen architecture or features changed')
        kernel, bias = feature.get_layer('classifier').get_weights()
        kernel[:,1:] += (delta[:128]/v['scale'][:,None]).astype(np.float32)
        bias[1:] += (delta[128]-np.sum(delta[:128]*v['mean'][:,None]/v['scale'][:,None],axis=0)).astype(np.float32)
        for actual, expected in zip(model.get_layer('classifier').get_weights(), (kernel,bias), strict=True):
            np.testing.assert_array_equal(actual, expected)
        after = model(v['x'], training=False).numpy()
        probes = model(data['probes'], training=False).numpy()
        expected = v['current'].astype(float).copy()
        expected[:,1:] += v['design']@delta
        difference = float(np.max(np.abs(after-expected)))
        if (difference>1e-4 or abs(difference-row['fold_max_delta'])>1e-8
                or abs(np.abs(delta).sum()-row['l1_head_change'])>1e-8
                or not (after[2074:2087,0,None]-after[2074:2087,1:] >= .001-5e-5).all()):
            raise ValueError('Folded head or new background bounds failed')
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            for k, value in dict(reference=v['reference'], feature=v['current'], after=after,
                                 probes=probes, labels=data['y'][:2074]).items():
                np.testing.assert_allclose(stored[k], value, atol=1e-5, rtol=0)
        measured = calculate(v['reference'][:2074], after[:2074], data['y'][:2074], probes, data['probe_labels'])
        measured['scene_decisions_retained'] = bool(np.array_equal(decisions(v['reference'][2087:]), decisions(after[2087:])))
        measured['training_gate_pass'] &= measured['scene_decisions_retained']
        if any(row[k] != value for k,value in measured.items()):
            raise ValueError('TRAIN gate does not reproduce')
        print(dict(audited=name, gate=measured['training_gate_pass'], new_correct=sum(measured['probe_correct']),
                   lost=len(measured['lost_old_rows'])), flush=True)
    checked_fit()
    passed = all(fit['results'][f'admit{s}']['training_gate_pass'] for s in (47,48))
    write_json(OUTPUT, dict(results=fit['results'], fit_sha256=sha256(FIT), protocol_sha256=sha256(PROTOCOL),
        script_sha256=sha256(__file__), eligible_for_full_scene_comparison=passed,
        scope='Reproduced TRAIN inputs, shifted reference bounds, convex certificates, unchanged features/background head, folded weights and checkpoint outputs. Both admission seeds required; no qualification.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
