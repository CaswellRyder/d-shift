"""Paired known-target coverage and FIT spatial diagnostics; not qualification."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.balloon_fine_grid import PROTOCOL, checked_grid_data, decode
from scripts.audit_balloon_dense_proposals import audit as control_audit, supplement
from scripts.evaluate_balloon_edge_search import coverage
from scripts.research_balloon_search import experimental
from scripts.summarize_dense_public_trial import split_coverage
from scripts.summarize_balloon_relative_size import passes_training_gate


def checked_predictions(run, targets, folds):
    run = Path(run)
    p, report = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    protocol = read_json(PROTOCOL)
    if (read_json(run / 'status.json')['state'] != 'complete'
            or p['manifest_sha256'] != protocol['manifest_sha256']
            or p['protocol_sha256'] != sha256(PROTOCOL)
            or p['script_sha256'] != sha256('scripts/train_balloon_fine_grid.py')
            or p['grid_helper_sha256'] != sha256('scripts/balloon_fine_grid.py')
            or any(sha256(path) != digest for path, digest in p['helpers_sha256'].items())
            or p['grid'] != [30, 40] or p['parameters'] != 2342
            or p['fixed_final_epoch'] is not True or p['folds'] != folds.tolist()
            or any(p[k] != protocol[k] for k in ('epochs', 'batch_size', 'learning_rate', 'box_objective'))
            or p['targets_sha256'] != sha256(run / 'targets.npy')
            or report['provenance_sha256'] != sha256(run / 'provenance.json')
            or report['model_sha256'] != sha256(run / 'proposal.keras')
            or report['predictions_sha256'] != sha256(run / 'predictions.npy')):
        raise ValueError('Fine-grid run identity changed')
    if not np.array_equal(targets, np.load(run / 'targets.npy', allow_pickle=False)):
        raise ValueError('Re-encoded targets changed')
    values = np.load(run / 'predictions.npy', allow_pickle=False)
    if values.shape != (len(targets), 30, 40, 6) or not np.isfinite(values).all():
        raise ValueError('Invalid saved outputs')
    return p, report, values


def spatial_diagnostic(doc, folds):
    result = {}
    for name, indices in (
        ('original_fit', [i for i, f in enumerate(folds) if f != 0 and i < 108]),
        ('public_fit', [i for i, f in enumerate(folds) if f != 0 and i >= 108]),
    ):
        grids = {}
        for step in (16, 8):
            collision, adjacent = Counter(), Counter()
            sizes = {c: [] for c in ('red_balloon', 'blue_balloon')}
            for i in indices:
                rows = doc['frames'][i]['truth']
                cells = [(int((r['box'][0] + r['box'][2]) / 2 / step),
                          int((r['box'][1] + r['box'][3]) / 2 / step)) for r in rows]
                for j, row in enumerate(rows):
                    if cells.count(cells[j]) > 1:
                        collision[row['label']] += 1
                    if any(k != j and q['label'] == row['label']
                           and max(abs(cells[k][0] - cells[j][0]), abs(cells[k][1] - cells[j][1])) <= 1
                           for k, q in enumerate(rows)):
                        adjacent[row['label']] += 1
                    a, b, c, d = row['box']
                    sizes[row['label']].append(min(c - a, d - b) / 2)
            grids[str(step)] = dict(colliding_targets=dict(collision), same_color_adjacent_targets=dict(adjacent))
        result[name] = dict(scene_cell_pitch=grids, input_short_side={c: dict(targets=len(v),
            quantiles=np.quantile(v, [0, .25, .5, .75, 1]).tolist(),
            below_four_pixels=sum(x < 4 for x in v), below_eight_pixels=sum(x < 8 for x in v)) for c, v in sizes.items()})
    return result


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = read_json(PROTOCOL)
    doc, _, targets, folds, _ = checked_grid_data()
    root = Path(protocol['data'])
    trials = {}
    for seed in protocol['seeds']:
        run = Path(f'runs/balloon-fine-grid-seed{seed}-20261011')
        p, report, values = checked_predictions(run, targets, folds)
        control_run = Path(protocol['controls'][str(seed)])
        if p['seed'] != seed or p['control_provenance_sha256'] != sha256(control_run / 'provenance.json'):
            raise ValueError('Paired control identity changed')
        control = control_audit(root, control_run)
        frames = []
        for row, value, fold in zip(doc['frames'], values, folds, strict=True):
            with Image.open(root / row['path']) as image:
                rgb = np.asarray(image.convert('RGB'))
            baseline, learned = experimental(rgb, 'mser'), decode(value)
            added = supplement(baseline, learned)
            frames.append(dict(source=row['source'], frame_sha256=row['sha256'], fold=int(fold),
                truth=row['truth'], coverage={k: coverage(row['truth'], proposals)
                    for k, proposals in (('baseline', baseline), ('learned', learned), ('supplement', added))},
                learned_proposals=learned, extra_calls=len(added) - len(baseline)))
        before, after = split_coverage(control['frames'], 108), split_coverage(frames, 108)
        for a, b in zip(frames, control['frames'], strict=True):
            if (a['source'] != b['source'] or a['truth'] != b['truth']
                    or a['coverage']['baseline'] != b['coverage']['baseline']):
                raise ValueError('Source annotations or baseline search changed')
        trials[str(seed)] = dict(control=before, fine_grid=after,
            training_gate_pass=passes_training_gate(before, after), frames=frames,
            model_sha256=report['model_sha256'], provenance_sha256=report['provenance_sha256'],
            predictions_sha256=report['predictions_sha256'], targets_sha256=p['targets_sha256'])
    passed = all(r['training_gate_pass'] for r in trials.values())
    result = dict(trials=trials, both_seeds_training_gate_pass=passed,
        spatial_diagnostic=spatial_diagnostic(doc, folds), omitted_negatives=doc['fine_grid_omitted_negatives'],
        decision='Eligible for fixed development evaluation' if passed else 'Reject before development; keep Pi unchanged',
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/balloon_fine_grid.py', 'scripts/summarize_dense_public_trial.py',
            'scripts/summarize_balloon_relative_size.py', 'scripts/evaluate_balloon_edge_search.py')},
        scope='Known TRAIN proposal coverage only, not full-scene precision; grouped holdout remains known-environment diagnostic and is not used by the training gate',
        development_evaluated=False, test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print({s: {k: v for k, v in r.items() if k != 'frames'} for s, r in trials.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
