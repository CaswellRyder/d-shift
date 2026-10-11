"""Reconstruct head constraints, folded inference and all TRAIN scene decisions."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.fit_batch5_head_repair import OUTPUT as FIT, PROTOCOL, prepared
from scripts.batch5_occlusion_data import tensors, new_scene_gate
from scripts.balloon_minimum_drift import certificate
from scripts.train_batch4_head_only import frozen_weights
from scripts.balloon_scene_constraints import decisions
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.train_batch3_retraining import calculate, correct
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/batch5-head-repair-audit-20261011.json')


def checked_fit():
    fit = read_json(FIT)
    if (any(sha256(p) != h for p, h in fit['sources_sha256'].items())
            or fit['protocol'] != read_json(PROTOCOL) or set(fit['results']) != {'control', 'hard'}
            or sha256(fit['raw']['path']) != fit['raw']['sha256']
            or any(fit[k] is not False for k in ('development_evaluated', 'test_evaluated', 'pi_tested',
                                               'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Fit provenance or scope changed')
    for row in fit['results'].values():
        root = Path(row['run'])
        if (read_json(root/'status.json')['state'] != 'complete'
                or sha256(root/'report.json') != row['report_sha256']
                or sha256(root/'constraints.npz') != row['archive_sha256']
                or (row['model_saved'] and sha256(root/'student.keras') != row['model_sha256'])):
            raise ValueError('Fit artifacts incomplete or changed')
    return fit


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    fit, data = checked_fit(), tensors('control')
    raw = read_json(fit['raw']['path'])
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(fit['protocol']['parent'], compile=False)
    results = {}
    for arm, row in fit['results'].items():
        root, v = Path(row['run']), prepared(data, arm, parent, keras)
        with np.load(root/'constraints.npz', allow_pickle=False) as archive:
            for key in ('x', 'design', 'logits', 'a', 'b', 'identities', 'h', 'mean', 'scale', 'objective_indices'):
                np.testing.assert_allclose(archive[key], v[key], atol=1e-8, rtol=0)
            if not row['model_saved']:
                if row['training_gate_pass']:
                    raise ValueError('Missing model claimed a pass')
                results[arm] = dict(model_saved=False, training_gate_pass=False, solver=row['solver'])
                continue
            delta = archive['delta']
            proof = certificate(v['h'], v['a'], v['b'], delta)
            for key in ('maximum_violation', 'stationarity', 'complementarity', 'objective'):
                if abs(proof[key]-row['solver'][key]) > 1e-8:
                    raise ValueError('Convex certificate does not reproduce')
            if proof['maximum_violation'] > 1e-6 or max(proof['stationarity'], proof['complementarity']) > 1e-4:
                raise ValueError('Invalid constrained optimum')
            model = keras.models.load_model(root/'student.keras', compile=False)
            if (model.count_params() != 7763 or model.get_config() != parent.get_config()
                    or not all(np.array_equal(a, b) for a, b in zip(frozen_weights(model), frozen_weights(parent), strict=True))):
                raise ValueError('Inference architecture or frozen features changed')
            mk, mb = model.get_layer('classifier').get_weights()
            pk, pb = parent.get_layer('classifier').get_weights()
            if not np.array_equal(mk[:, 0], pk[:, 0]) or mb[0] != pb[0]:
                raise ValueError('Background head changed')
            actual = model(v['x'], training=False).numpy()
            np.testing.assert_allclose(archive['after'], actual, atol=1e-5, rtol=0)
        expected = v['logits'].astype(float).copy()
        expected[:, 1:] += v['design']@delta.reshape(129, 2)
        error = float(np.max(np.abs(actual-expected)))
        if error > 1e-4:
            raise ValueError('Folded model differs from solved logits')
        scene_indices = v['objective_indices'][v['objective_indices'] >= 2094]
        if not np.array_equal(decisions(actual[scene_indices]), decisions(v['logits'][scene_indices])):
            raise ValueError('A protected proposal decision changed')
        found = dict(original=replay(raw['parent']['original']['frames'], softmax(actual[2094:2479])),
                     batch5=replay(raw['parent']['batch5']['frames'], softmax(actual[2479:])))
        for kind, result in found.items():
            if checked_metrics(result) != row[kind+'_metrics'] or result != raw[arm][kind]:
                raise ValueError('Scene replay/counts changed')
            baseline_scores = v['logits'][2094:2479] if kind == 'original' else v['logits'][2479:]
            baseline = replay(raw['parent'][kind]['frames'], softmax(baseline_scores))
            for a, b in zip(baseline['frames'], raw['parent'][kind]['frames'], strict=True):
                check_replayed_parent(a, b)
        effect = difference(found['original'], raw['parent']['original'])
        negatives = [false_acceptances(z[2074:2087]) for z in (v['logits'], actual)]
        crop = calculate(v['logits'][:2074], actual[:2074], data['y'][:2074], actual[2074:2087], data['probe_labels'])
        crop_retained = not crop['lost_old_rows'] and all(r['after'] <= r['before'] for r in crop['false_acceptances'].values())
        hard_logits = actual[v['exempt']:v['exempt']+1]
        hard_ok = bool(correct(hard_logits, np.array([2]))[0])
        gate = bool(crop_retained and scene_gate(effect, *negatives) and new_scene_gate(found['batch5'], hard_ok))
        if (gate != row['training_gate_pass'] or crop != row['crop_diagnostics']
                or negatives != row['background_false_acceptances'] or effect != row['effect']):
            raise ValueError('Predeclared gate does not reproduce')
        results[arm] = dict(run=str(root), model_sha256=row['model_sha256'], training_gate_pass=gate,
            original_metrics=found['original']['metrics'], batch5_metrics=found['batch5']['metrics'],
            hard_blue_probability=float(softmax(hard_logits)[0, 2]), protected_scene_decisions_retained=True,
            old_crop_lost=len(crop['lost_old_rows']), fold_max_delta=error, certificate=proof)
        print(dict(arm=arm, **results[arm]), flush=True)
    eligible = bool(results['hard']['training_gate_pass'])
    if eligible != fit['eligible_for_independent_audit']:
        raise ValueError('Audited eligibility changed')
    checked_fit()
    write_json(OUTPUT, dict(results=results, fit_sha256=sha256(FIT), script_sha256=sha256(__file__),
        helpers_sha256={p:sha256(p) for p in ('scripts/batch5_head_constraints.py',
            'scripts/balloon_minimum_drift.py', 'scripts/summarize_public_balloon_trial.py')},
        eligible_for_export_and_development=eligible,
        scope='Independent reconstruction of TRAIN constraints, KKT certificate, frozen weights, folded predictions and scene matching. Not held-out accuracy or Pi readiness.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
