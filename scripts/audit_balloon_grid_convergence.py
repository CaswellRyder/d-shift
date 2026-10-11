"""Verify fixed continuation and compare known-target coverage without threshold tuning."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.train_balloon_grid_convergence import PROTOCOL, source_data
from scripts.audit_balloon_dense_proposals import audit as coarse_audit, decode as coarse_decode, supplement
from scripts.balloon_fine_grid import decode as fine_decode
from scripts.evaluate_balloon_edge_search import coverage
from scripts.research_balloon_search import experimental
from scripts.summarize_dense_public_trial import split_coverage
from scripts.summarize_balloon_relative_size import passes_training_gate


def checked_run(arm, seed):
    protocol = read_json(PROTOCOL)
    doc, x, targets, folds, groups = source_data(arm)
    run = Path(f'runs/balloon-grid-convergence-{arm}-seed{seed}-20261011')
    p, report = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    parent = Path(f"runs/{protocol['parent_prefix'][arm]}-seed{seed}-20261011")
    steps = math.ceil(int((folds != 0).sum()) / protocol['batch_size'])
    if (read_json(run / 'status.json')['state'] != 'complete'
            or p['arm'] != arm or p['seed'] != seed or p['parent'] != str(parent)
            or p['parent_model_sha256'] != sha256(parent / 'proposal.keras')
            or p['parent_provenance_sha256'] != sha256(parent / 'provenance.json')
            or p['parent_report_sha256'] != sha256(parent / 'report.json')
            or p['manifest_sha256'] != protocol['manifest_sha256']
            or p['protocol_sha256'] != sha256(PROTOCOL)
            or p['script_sha256'] != sha256('scripts/train_balloon_grid_convergence.py')
            or any(sha256(path) != digest for path, digest in p['helpers_sha256'].items())
            or p['parameters'] != 2342 or p['fixed_final_epoch'] is not True
            or p['folds'] != folds.tolist() or p['groups'] != groups
            or p['epochs'] != protocol['final_epochs'] or p['initial_epoch'] != protocol['parent_epochs']
            or p['optimizer_restored']['iterations'] != steps * protocol['parent_epochs']
            or report['optimizer_final']['iterations'] != steps * protocol['final_epochs']
            or any(p[k] != protocol[k] for k in ('batch_size', 'learning_rate'))
            or p['targets_sha256'] != sha256(run / 'targets.npy')
            or report['provenance_sha256'] != sha256(run / 'provenance.json')
            or report['model_sha256'] != sha256(run / 'proposal.keras')
            or report['predictions_sha256'] != sha256(run / 'predictions.npy')):
        raise ValueError('Continuation identity changed')
    if not np.array_equal(targets, np.load(run / 'targets.npy', allow_pickle=False)):
        raise ValueError('Supervision changed')
    values = np.load(run / 'predictions.npy', allow_pickle=False)
    grid = (15, 20) if arm == 'coarse' else (30, 40)
    if values.shape != (len(targets), *grid, 6) or not np.isfinite(values).all():
        raise ValueError('Invalid outputs')
    return run, doc, x, targets, folds, p, report, values


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = read_json(PROTOCOL)
    root = Path(protocol['data'])
    controls = {str(s): coarse_audit(root, f'runs/balloon-dense-public-admit-seed{s}-20261011') for s in protocol['seeds']}
    trials = {}
    for arm in protocol['arms']:
        trials[arm] = {}
        decoder = coarse_decode if arm == 'coarse' else fine_decode
        for seed in protocol['seeds']:
            run, doc, _, _, folds, p, report, values = checked_run(arm, seed)
            frames = []
            for row, value, fold in zip(doc['frames'], values, folds, strict=True):
                with Image.open(root / row['path']) as image:
                    rgb = np.asarray(image.convert('RGB'))
                baseline, learned = experimental(rgb, 'mser'), decoder(value)
                added = supplement(baseline, learned)
                frames.append(dict(source=row['source'], frame_sha256=row['sha256'], fold=int(fold),
                    truth=row['truth'], coverage={k: coverage(row['truth'], proposals)
                        for k, proposals in (('baseline', baseline), ('learned', learned), ('supplement', added))},
                    learned_proposals=learned, extra_calls=len(added) - len(baseline)))
            control = controls[str(seed)]
            for a, b in zip(frames, control['frames'], strict=True):
                if (a['source'] != b['source'] or a['truth'] != b['truth']
                        or a['coverage']['baseline'] != b['coverage']['baseline']):
                    raise ValueError('Fixed baseline or labels changed')
            before, after = split_coverage(control['frames'], 108), split_coverage(frames, 108)
            trials[arm][str(seed)] = dict(control_coarse80=before, continued=after,
                training_gate_pass=passes_training_gate(before, after), frames=frames,
                model_sha256=report['model_sha256'], provenance_sha256=report['provenance_sha256'],
                predictions_sha256=report['predictions_sha256'], report=report)
    eligible = [arm for arm, seeds in trials.items() if all(r['training_gate_pass'] for r in seeds.values())]
    result = dict(trials=trials, eligible_arms=eligible, protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        scope='Known TRAIN target coverage only; unknown objects are not labelled. Grouped holdout is not independent qualification or used for selection.',
        development_evaluated=False, test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(dict(eligible_arms=eligible, trials={a: {s: dict(continued=r['continued'], training_gate_pass=r['training_gate_pass'])
        for s, r in seeds.items()} for a, seeds in trials.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
