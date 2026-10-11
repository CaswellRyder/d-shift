"""Compare paired size-loss runs without using development to select a model."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_dense_proposals import audit
from scripts.summarize_dense_public_trial import split_coverage
from scripts.train_balloon_relative_size import PROTOCOL


def passes_training_gate(control, candidate):
    for split in ('common_fit', 'extra_fit'):
        for label in ('red_balloon', 'blue_balloon'):
            before, after = control[split][label], candidate[split][label]
            if before['targets'] != after['targets'] or after['learned'] < before['learned']:
                return False
    return candidate['extra_fit']['blue_balloon']['learned'] > control['extra_fit']['blue_balloon']['learned']


def summarize(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = read_json(PROTOCOL)
    root = Path(protocol['data'])
    if sha256(root / 'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Data identity changed')
    trials = {}
    for seed in protocol['seeds']:
        arms = {}
        for mode, run in (
            ('control', Path(protocol['controls'][str(seed)])),
            ('relative_size', Path(f'runs/balloon-relative-size-seed{seed}-20261011')),
        ):
            p = read_json(run / 'provenance.json')
            if (p['seed'] != seed or p['parameters'] != 2342 or p['fixed_final_epoch'] is not True
                    or any(p[k] != protocol[k] for k in ('epochs', 'batch_size', 'learning_rate'))):
                raise ValueError('Paired schedule changed')
            if mode == 'relative_size' and (
                p['protocol_sha256'] != sha256(PROTOCOL)
                or p['script_sha256'] != sha256('scripts/train_balloon_relative_size.py')
                or p['architecture_helper_sha256'] != sha256('scripts/train_balloon_dense_proposals.py')
                or p['loss_helper_sha256'] != sha256('scripts/balloon_dense_iou_loss.py')
                or p['size_weight'] != 1. or p['size_epsilon'] != 1e-4
                or p['control_provenance_sha256'] != sha256(Path(protocol['controls'][str(seed)]) / 'provenance.json')
            ):
                raise ValueError('Objective or control changed')
            coverage = audit(root, run)
            # The first 108 rows are the frozen original dataset. Validate the
            # boundary against derivative metadata instead of merely slicing.
            manifest = read_json(root / 'manifest.json')
            if (len(manifest['frames']) != 129
                    or any(r.get('expansion_kind') == 'public' for r in manifest['frames'][:108])
                    or any(r.get('expansion_kind') != 'public' for r in manifest['frames'][108:])):
                raise ValueError('Original/public boundary changed')
            arms[mode] = dict(run=str(run), coverage=split_coverage(coverage['frames'], 108),
                model_sha256=sha256(run / 'proposal.keras'),
                provenance_sha256=sha256(run / 'provenance.json'),
                predictions_sha256=sha256(run / 'predictions.npy'))
        gate = passes_training_gate(arms['control']['coverage'], arms['relative_size']['coverage'])
        trials[str(seed)] = dict(arms=arms, training_gate_pass=gate)
    passed = all(r['training_gate_pass'] for r in trials.values())
    result = dict(trials=trials, both_seeds_training_gate_pass=passed,
        decision='Eligible for fixed development evaluation' if passed else 'Reject before development; keep Pi unchanged',
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        scope='Known TRAIN proposal coverage only; no full-scene precision. Common grouped holdout is descriptive and not used by the gate.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    summarize(p.parse_args().output)
