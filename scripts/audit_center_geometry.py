"""Paired TRAIN center/geometry gate with frozen-parent and checkpoint checks."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.grid_blue_expansion_data import expected, BASE
from scripts.train_grid_blue_expansion import tensor_hash
from scripts.train_center_geometry import PROTOCOL, run_path
from scripts.balloon_center_geometry_adapter import frozen_weights, joint_loss
from scripts.audit_grid_blue_expansion import coverage_report
from scripts.train_balloon_grid_convergence import optimizer_evidence
from scripts.research_balloon_search import experimental

TRIAL = Path('research/balloon-red-blue-20261008/center-geometry-training-20261011.json')
FRAMES = Path('runs/center-geometry-training-frames-20261011.json')


def gate(parent, current, control):
    colors = ('red_balloon', 'blue_balloon')
    for split in parent:
        for color in colors:
            if len({v[split][color]['targets'] for v in (parent, current, control)}) != 1:
                raise ValueError('Coverage target count changed')
    return (all(current['base_fit'][color][method] >= parent['base_fit'][color][method]
                for color in colors for method in ('learned', 'supplement'))
            and current['extra_fit']['blue_balloon']['learned'] > max(
                parent['extra_fit']['blue_balloon']['learned'], control['extra_fit']['blue_balloon']['learned']))


def checked_run(mode, seed):
    settings = read_json(PROTOCOL)
    data = expected(mode)
    probe = expected('admit') if mode == 'control' else data
    run = run_path(mode, seed)
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete' or p['mode'] != mode or p['seed'] != seed
            or p['protocol_sha256'] != sha256(PROTOCOL) or p['parent'] != settings['parent']
            or p['parent_sha256'] != settings['parent_sha256'] or sha256(p['parent']) != p['parent_sha256']
            or p['records'] != data['records'] or p['probe_records'] != probe['records']
            or p['stats'] != data['stats'] or p['folds'] != data['folds'].tolist()
            or p['parameters'] != 3060 or p['trainable_parameters'] != 718
            or p['optimizer_reset'] is not True or p['fixed_final_epoch'] is not True
            or p['initial_parity_frames'] != 156 or not 0 <= p['initial_delta'] <= 1e-5
            or r['optimizer_iterations'] != 5760 or r['frozen_weights_unchanged'] is not True
            or r['model_sha256'] != sha256(run / 'proposal.keras')
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())):
        raise ValueError('Joint training identity changed')
    tensors = dict(train_images=data['images'], train_targets=data['targets'],
                   probe_images=probe['images'], probe_targets=probe['targets'])
    if p['tensors_sha256'] != {k: tensor_hash(v) for k, v in tensors.items()}:
        raise ValueError('Training or probe pixels/labels changed')
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    if set(arrays) != {'before', 'after'} or any(v.shape != (156, 15, 20, 6) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Invalid saved predictions')
    delta = float(np.max(np.abs(arrays['before'][..., :2] - arrays['after'][..., :2])))
    if delta != r['max_heat_delta']:
        raise ValueError('Recorded heat delta does not reproduce')
    return run, data, probe, p, r, arrays


def audit():
    if TRIAL.exists() or FRAMES.exists():
        raise FileExistsError('Do not overwrite geometry trial receipts')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    geometry_path = Path('research/balloon-red-blue-20261008/geometry-adapter-training-20261011.json')
    geometry_reference = read_json(geometry_path)
    if geometry_reference['script_sha256'] != sha256('scripts/audit_geometry_adapter.py'):
        raise ValueError('Prior geometry-only audit changed')
    parent = keras.models.load_model(read_json(PROTOCOL)['parent'], compile=False)
    trials, raw, baselines = {}, {}, None
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, data, probe, p, r, arrays = checked_run(mode, seed)
            model = keras.models.load_model(run / 'proposal.keras', custom_objects={'joint_loss': joint_loss})
            optimizer = optimizer_evidence(model, 5760)
            if model.count_params() != 3060 or frozen_weights(model) != p['frozen_weights']:
                raise ValueError('Frozen checkpoint weights changed')
            for loaded, key in ((parent, 'before'), (model, 'after')):
                if not np.allclose(loaded(probe['images'], training=False).numpy(), arrays[key], rtol=0, atol=1e-5):
                    raise ValueError('Checkpoint does not reproduce saved predictions')
            fit, hold = data['folds'] != 0, data['folds'] == 0
            losses = dict(training_loss=float(joint_loss(tf.constant(data['targets'][fit]), model(data['images'][fit], training=False))),
                          grouped_holdout_loss=float(joint_loss(tf.constant(data['targets'][hold]), tf.constant(arrays['after'][hold]))))
            if any(not np.isclose(r[k], v, rtol=0, atol=1e-5) for k, v in losses.items()):
                raise ValueError('Loss diagnostics do not reproduce')
            if baselines is None:
                baselines = []
                for i, row in enumerate(probe['records']):
                    path = BASE / row['path'] if i < 129 else Path(row['source_path'])
                    with Image.open(path) as image:
                        baselines.append(experimental(np.asarray(image.convert('RGB')), 'mser'))
            before, _ = coverage_report(probe, arrays['before'], baselines)
            after, frames = coverage_report(probe, arrays['after'], baselines)
            name = f'{mode}{seed}'
            trials[name] = dict(path=str(run), report=r, optimizer=optimizer, parent_coverage=before, coverage=after,
                                geometry_only_coverage=geometry_reference['trials'][name]['coverage'])
            raw[name] = frames
    for seed in (47, 48):
        trial = trials[f'admit{seed}']
        trial['training_gate_pass'] = gate(trial['parent_coverage'], trial['coverage'], trials[f'control{seed}']['coverage'])
    eligible = all(trials[f'admit{s}']['training_gate_pass'] for s in (47, 48))
    write_json(FRAMES, raw)
    write_json(TRIAL, dict(trials=trials, eligible_for_development=eligible,
        frames=dict(path=str(FRAMES), sha256=sha256(FRAMES)), protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        geometry_reference_sha256=sha256(geometry_path),
        helpers_sha256={p: sha256(p) for p in ('scripts/audit_grid_blue_expansion.py',
            'scripts/audit_balloon_dense_proposals.py', 'scripts/evaluate_balloon_edge_search.py',
            'scripts/research_balloon_search.py', 'scripts/train_balloon_grid_convergence.py')},
        scope='Known TRAIN joint center/geometry coverage only. Original grouped holdout is related and not used for selection. No full-scene precision or Pi timing claim.',
        test_evaluated=False, development_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible, trials={k: {field: v[field] for field in
        ('parent_coverage', 'coverage', 'training_gate_pass') if field in v} for k, v in trials.items()}))


if __name__ == '__main__':
    audit()
