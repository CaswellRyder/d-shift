"""Correct newly trained features against reference-parent TRAIN decision bounds."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.clutter_round3_feature_data import tensors, hashes
from scripts.balloon_rebased_constraints import system
from scripts.balloon_head_constraints import solve as feasible_solution
from scripts.balloon_minimum_drift import quadratic, solve
from scripts.balloon_scene_constraints import decisions
from scripts.train_batch3_retraining import calculate
from scripts.train_batch4_head_only import frozen_weights

PROTOCOL = Path('configs/clutter-feature-correction-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/clutter-feature-correction-20261011.json')


def checked_sources():
    p = read_json(PROTOCOL)
    audit = read_json(p['feature_audit'])
    if (set(audit['results']) != set(p['names']) or sha256(p['reference']) != p['reference_sha256']
            or audit['script_sha256'] != sha256('scripts/audit_clutter_round3_feature.py')
            or audit['data_helper_sha256'] != sha256('scripts/clutter_round3_feature_data.py')
            or audit['protocol_sha256'] != sha256('configs/clutter-round3-feature-20261011.json')
            or any(audit[k] is not False for k in ('development_evaluated', 'test_evaluated', 'pi_tested',
                                                  'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Feature training identity changed')
    for r in audit['results'].values():
        root = Path(r['run'])
        prov = read_json(root/'provenance.json')
        if (read_json(root/'status.json')['state'] != 'complete'
                or any(sha256(k) != h for k, h in prov['sources_sha256'].items())):
            raise ValueError('Feature training incomplete or changed')
        for field, file in (('model_sha256','student.keras'), ('predictions_sha256','predictions.npz'),
                            ('report_sha256','report.json'), ('provenance_sha256','provenance.json')):
            if sha256(root/file) != r[field]:
                raise ValueError('Feature checkpoint changed')
    return p, audit


def prepared(data, reference, feature, keras):
    old, chosen, scene = data['x'][:2074], data['x'][2074:2087], data['x'][-385:]
    x = np.concatenate([old, chosen, scene])
    extractor = keras.Model(feature.inputs[0], feature.get_layer('spatial_features').output)
    values = extractor(x, training=False).numpy()
    if values.shape != (2472, 128):
        raise ValueError('Feature dimensions changed')
    mean = values[:2074].astype(float).mean(0)
    scale = np.maximum(values[:2074].astype(float).std(0), 1e-6)
    design = np.column_stack([(values.astype(float)-mean)/scale, np.ones(len(x))])
    current = feature(x, training=False).numpy()
    ref = np.concatenate([reference(old, training=False).numpy(), current[2074:2087],
                          reference(scene, training=False).numpy()])
    a, b, identities = system(design, ref, current, data['y'][:2074])
    h = quadratic(np.concatenate([design[:2074], design[2087:]]))
    return dict(x=x, design=design, current=current, reference=ref, mean=mean, scale=scale,
                a=a, b=b, identities=identities, h=h)


def fold(model, delta, mean, scale):
    kernel, bias = model.get_layer('classifier').get_weights()
    kernel[:, 1:] += (delta[:128]/scale[:, None]).astype(np.float32)
    bias[1:] += (delta[128]-np.sum(delta[:128]*mean[:, None]/scale[:, None], axis=0)).astype(np.float32)
    model.get_layer('classifier').set_weights([kernel, bias])


def fit():
    p, audit = checked_sources()
    roots = {n: Path(f'runs/clutter-feature-correction-{n}-20261011') for n in p['names']}
    if OUTPUT.exists() or any(r.exists() for r in roots.values()):
        raise FileExistsError('Preserve existing correction experiment')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    reference = keras.models.load_model(p['reference'], compile=False)
    cache = {a: tensors(a) for a in ('control', 'admit')}
    paths = [PROTOCOL, p['feature_audit'], __file__, 'scripts/balloon_rebased_constraints.py',
        'scripts/balloon_minimum_drift.py', 'scripts/balloon_head_constraints.py',
        'scripts/balloon_scene_constraints.py', 'scripts/balloon_false_acceptance_constraints.py',
        'scripts/train_batch3_retraining.py', 'scripts/train_batch4_head_only.py',
        'scripts/clutter_round3_feature_data.py']
    sources = {str(k): sha256(k) for k in paths}
    results = {}
    for name, root in roots.items():
        data = cache['admit' if name.startswith('admit') else 'control']
        feature_path = Path(audit['results'][name]['run'])/'student.keras'
        feature = keras.models.load_model(feature_path, compile=False)
        if feature.count_params() != 7763:
            raise ValueError('Architecture changed')
        v = prepared(data, reference, feature, keras)
        root.mkdir(parents=True)
        write_json(root/'status.json', dict(state='running'))
        initial, feasibility = feasible_solution(v['a'], v['b'], time_limit=60.)
        delta, solver = solve(v['a'], v['b'], v['h'], initial, time_limit=120.) if initial is not None else (None, feasibility)
        stored = {k: v[k] for k in ('a', 'b', 'identities', 'h', 'mean', 'scale')}
        if initial is not None:
            stored['initial'] = initial
        report = dict(feature_path=str(feature_path), feature_sha256=sha256(feature_path),
            input_sha256=hashes(data), feasibility=feasibility, solver=solver,
            constraints=len(v['b']), model_saved=False, training_gate_pass=False)
        if delta is not None:
            delta = delta.reshape(129, 2)
            model = keras.models.load_model(feature_path, compile=False)
            fold(model, delta, v['mean'], v['scale'])
            if not all(np.array_equal(a, b) for a, b in zip(frozen_weights(feature), frozen_weights(model), strict=True)):
                raise ValueError('Frozen features changed')
            after = model(v['x'], training=False).numpy()
            expected = v['current'].astype(float).copy()
            expected[:, 1:] += v['design']@delta
            difference = float(np.max(np.abs(after-expected)))
            if not np.isfinite(after).all() or difference > 1e-4:
                raise ValueError('Folded solution differs')
            probes = model(data['probes'], training=False).numpy()
            report.update(calculate(v['reference'][:2074], after[:2074], data['y'][:2074], probes, data['probe_labels']))
            report['scene_decisions_retained'] = bool(np.array_equal(decisions(v['reference'][2087:]), decisions(after[2087:])))
            report['training_gate_pass'] &= report['scene_decisions_retained']
            report.update(fold_max_delta=difference, l1_head_change=float(np.abs(delta).sum()))
            model.save(root/'student.keras')
            np.savez_compressed(root/'predictions.npz', reference=v['reference'], feature=v['current'],
                                after=after, probes=probes, labels=data['y'][:2074])
            stored['delta'] = delta
            report.update(model_saved=True, model_sha256=sha256(root/'student.keras'),
                          predictions_sha256=sha256(root/'predictions.npz'))
        np.savez_compressed(root/'constraints.npz', **stored)
        report['constraint_archive_sha256'] = sha256(root/'constraints.npz')
        write_json(root/'report.json', report)
        write_json(root/'status.json', dict(state='complete'))
        results[name] = dict(run=str(root), report_sha256=sha256(root/'report.json'), **report)
        print(dict(name=name, solved=solver['success'], gate=report['training_gate_pass'],
            new_correct=sum(report.get('probe_correct', [])), lost=len(report.get('lost_old_rows', [])),
            objective=solver.get('objective')), flush=True)
    checked_sources()
    if any(sha256(k) != h for k, h in sources.items()):
        raise ValueError('Source changed during fitting')
    write_json(OUTPUT, dict(results=results, sources_sha256=sources, updated_parameters=258,
        total_parameters=7763, scope='TRAIN-only minimum-drift head correction on newly learned features. Original reference-parent retention bounds, matched controls and both seeds. No scene accuracy or hardware claim.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    fit()
