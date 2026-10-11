"""Compare final context-grid checkpoints with paired fine and coarse controls."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.balloon_context_grid import PROTOCOL, spatial_cost, network
from scripts.balloon_fine_grid import checked_grid_data, decode, network as fine_network
from scripts.audit_balloon_fine_grid import checked_predictions as fine_predictions
from scripts.audit_balloon_dense_proposals import audit as coarse_audit, supplement
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
            or p['script_sha256'] != sha256('scripts/balloon_context_grid.py')
            or any(sha256(path) != digest for path, digest in p['helpers_sha256'].items())
            or p['architecture']['parameters'] != 2342 or p['fixed_final_epoch'] is not True
            or p['folds'] != folds.tolist()
            or any(p[k] != protocol[k] for k in ('epochs', 'batch_size', 'learning_rate', 'box_objective'))
            or p['targets_sha256'] != sha256(run / 'targets.npy')
            or report['provenance_sha256'] != sha256(run / 'provenance.json')
            or report['model_sha256'] != sha256(run / 'proposal.keras')
            or report['predictions_sha256'] != sha256(run / 'predictions.npy')):
        raise ValueError('Context-grid identity changed')
    if not np.array_equal(targets, np.load(run / 'targets.npy', allow_pickle=False)):
        raise ValueError('Paired targets changed')
    values = np.load(run / 'predictions.npy', allow_pickle=False)
    if values.shape != (len(targets), 30, 40, 6) or not np.isfinite(values).all():
        raise ValueError('Invalid outputs')
    return p, report, values


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = read_json(PROTOCOL)
    doc, _, targets, folds, _ = checked_grid_data()
    if sha256(Path(protocol['data']) / 'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Data identity changed')
    root = Path(protocol['data'])
    trials = {}
    for seed in protocol['seeds']:
        run = Path(f'runs/balloon-context-grid-seed{seed}-20261011')
        p, report, values = checked_predictions(run, targets, folds)
        fine_run = Path(protocol['fine_controls'][str(seed)])
        coarse_run = Path(protocol['coarse_controls'][str(seed)])
        fine_p, fine_report, fine_values = fine_predictions(fine_run, targets, folds)
        if (p['seed'] != seed or fine_p['seed'] != seed
                or p['fine_control_provenance_sha256'] != sha256(fine_run / 'provenance.json')):
            raise ValueError('Fine control identity changed')
        coarse = coarse_audit(root, coarse_run)
        arms = {}
        for name, predictions in (('fine', fine_values), ('context', values)):
            frames = []
            for row, value, fold in zip(doc['frames'], predictions, folds, strict=True):
                with Image.open(root / row['path']) as image:
                    rgb = np.asarray(image.convert('RGB'))
                baseline, learned = experimental(rgb, 'mser'), decode(value)
                added = supplement(baseline, learned)
                frames.append(dict(source=row['source'], frame_sha256=row['sha256'], fold=int(fold),
                    truth=row['truth'], coverage={k: coverage(row['truth'], proposals)
                        for k, proposals in (('baseline', baseline), ('learned', learned), ('supplement', added))},
                    learned_proposals=learned, extra_calls=len(added) - len(baseline)))
            for a, b in zip(frames, coarse['frames'], strict=True):
                if (a['source'] != b['source'] or a['truth'] != b['truth']
                        or a['coverage']['baseline'] != b['coverage']['baseline']):
                    raise ValueError('Baseline or source annotations changed')
            arms[name] = dict(coverage=split_coverage(frames, 108), frames=frames)
        old = split_coverage(coarse['frames'], 108)
        trials[str(seed)] = dict(coarse=old, **arms,
            training_gate_pass=passes_training_gate(old, arms['context']['coverage']),
            model_sha256=report['model_sha256'], provenance_sha256=report['provenance_sha256'],
            predictions_sha256=report['predictions_sha256'], fine_model_sha256=fine_report['model_sha256'])
    from scripts.train_balloon_dense_proposals import network as coarse_network
    architecture = {k: spatial_cost(n()) for k, n in (('coarse', coarse_network), ('fine', fine_network), ('context', network))}
    passed = all(r['training_gate_pass'] for r in trials.values())
    result = dict(trials=trials, architecture=architecture, both_seeds_training_gate_pass=passed,
        decision='Eligible for fixed development evaluation' if passed else 'Reject before development; keep Pi unchanged',
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/balloon_fine_grid.py', 'scripts/summarize_dense_public_trial.py',
            'scripts/summarize_balloon_relative_size.py', 'scripts/evaluate_balloon_edge_search.py')},
        scope='Known TRAIN proposal coverage, not full-scene precision or qualification; common grouped holdout is descriptive only',
        development_evaluated=False, test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(dict(architecture=architecture, trials={s: dict(coarse=r['coarse'], fine=r['fine']['coverage'],
        context=r['context']['coverage'], training_gate_pass=r['training_gate_pass']) for s, r in trials.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
