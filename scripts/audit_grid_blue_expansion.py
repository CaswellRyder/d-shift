"""Verify saved proposal adaptation and apply its predeclared TRAIN-only gate."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.grid_blue_expansion_data import expected, BASE
from scripts.train_grid_blue_expansion import PROTOCOL, run_path, tensor_hash
from scripts.audit_balloon_dense_proposals import decode, supplement
from scripts.evaluate_balloon_edge_search import coverage
from scripts.research_balloon_search import experimental

OUTPUT = Path('research/balloon-red-blue-20261008/grid-blue-expansion-training-20261011.json')
LABELS = ('red_balloon', 'blue_balloon')


def gate(before, after, negative_before, negative_after):
    if not np.isfinite([negative_before, negative_after]).all():
        raise ValueError('Nonfinite background diagnostic')
    for split in before:
        for color in LABELS:
            if before[split][color]['targets'] != after[split][color]['targets']:
                raise ValueError('Coverage target count changed')
    return (all(after['base_fit'][c]['learned'] >= before['base_fit'][c]['learned'] for c in LABELS)
            and after['extra_fit']['blue_balloon']['learned'] > before['extra_fit']['blue_balloon']['learned']
            and negative_after < negative_before)


def checked_run(mode, seed):
    protocol = read_json(PROTOCOL)
    data = expected(mode)
    probe = expected('admit') if mode == 'control' else data
    run = run_path(mode, seed)
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete' or p['mode'] != mode or p['seed'] != seed
            or p['protocol_sha256'] != sha256(PROTOCOL) or p['parent'] != protocol['parent']
            or p['parent_sha256'] != protocol['parent_sha256'] or sha256(p['parent']) != p['parent_sha256']
            or p['records'] != data['records'] or p['probe_records'] != probe['records']
            or p['stats'] != data['stats'] or p['folds'] != data['folds'].tolist()
            or p['parameters'] != 2342 or p['optimizer_reset'] is not True or p['fixed_final_epoch'] is not True
            or r['optimizer_iterations'] != 2880
            or r['model_sha256'] != sha256(run / 'proposal.keras')
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != digest for path, digest in p['sources_sha256'].items())):
        raise ValueError('Training identity changed')
    actual = {k: tensor_hash(v) for k, v in dict(train_images=data['images'], train_targets=data['targets'],
        probe_images=probe['images'], probe_targets=probe['targets']).items()}
    if actual != p['tensors_sha256']:
        raise ValueError('Training or probe pixels/supervision changed')
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    if set(arrays) != {'before', 'after'} or any(v.shape != (156, 15, 20, 6) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Invalid saved predictions')
    return run, data, probe, p, r, arrays


def negative_heat(probe, values):
    mask = np.zeros(values.shape[:-1], bool)
    for i, row in enumerate(probe['records']):
        if i >= 129 and not row['truth']:
            mask[i] = probe['targets'][i, ..., 3] > 0
    if not mask.any():
        raise ValueError('No reviewed negative-only heat cells')
    logits = values[..., 1][mask]
    return float(np.mean(1 / (1 + np.exp(-np.clip(logits, -80, 80))))), int(mask.sum())


def coverage_report(probe, values, baselines):
    frames = []
    for i, (row, value, fold, base) in enumerate(zip(probe['records'], values, probe['folds'], baselines, strict=True)):
        learned = decode(value)
        added = supplement(base, learned)
        frames.append(dict(source=row['source'], truth=row['truth'], fold=int(fold),
            split='extra_fit' if i >= 129 else ('base_holdout' if fold == 0 else 'base_fit'),
            coverage={name: coverage(row['truth'], proposals) for name, proposals in
                (('baseline', base), ('learned', learned), ('supplement', added))},
            learned_proposals=learned, extra_calls=len(added) - len(base)))
    totals = {}
    for split in ('base_fit', 'base_holdout', 'extra_fit'):
        subset = [r for r in frames if r['split'] == split]
        totals[split] = {color: dict(targets=sum(r['coverage']['baseline'][color]['targets'] for r in subset),
            **{name: sum(r['coverage'][name][color]['localized'] for r in subset)
                for name in ('baseline', 'learned', 'supplement')}) for color in LABELS}
    return totals, frames


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(read_json(PROTOCOL)['parent'], compile=False)
    trials, baselines = {}, None
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, _, probe, p, r, arrays = checked_run(mode, seed)
            model = keras.models.load_model(run / 'proposal.keras', compile=False)
            for loaded, key in ((parent, 'before'), (model, 'after')):
                if not np.allclose(loaded(probe['images'], training=False).numpy(), arrays[key], rtol=0, atol=1e-5):
                    raise ValueError('Saved checkpoint predictions do not reproduce')
            if baselines is None:
                baselines = []
                for i, row in enumerate(probe['records']):
                    path = BASE / row['path'] if i < 129 else Path(row['source_path'])
                    with Image.open(path) as image:
                        baselines.append(experimental(np.asarray(image.convert('RGB')), 'mser'))
            before, _ = coverage_report(probe, arrays['before'], baselines)
            after, frames = coverage_report(probe, arrays['after'], baselines)
            old_heat, cells = negative_heat(probe, arrays['before'])
            new_heat, _ = negative_heat(probe, arrays['after'])
            trials[f'{mode}{seed}'] = dict(path=str(run), report=r, parent_coverage=before,
                coverage=after, frames=frames, negative_heat_before=old_heat, negative_heat_after=new_heat,
                negative_heat_cells=cells, training_gate_pass=gate(before, after, old_heat, new_heat))
    eligible = all(trials[f'admit{s}']['training_gate_pass'] for s in (47, 48))
    write_json(OUTPUT, dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/audit_balloon_dense_proposals.py',
            'scripts/evaluate_balloon_edge_search.py', 'scripts/research_balloon_search.py')},
        scope='Known TRAIN box coverage only; unknown regions are not negatives. Grouped holdout diagnostic is related and not used for model selection. No scene precision, live timing or qualification claim.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible, trials={k: {field: v[field] for field in
        ('parent_coverage', 'coverage', 'negative_heat_before', 'negative_heat_after', 'training_gate_pass')}
        for k, v in trials.items()}))


if __name__ == '__main__':
    audit()
