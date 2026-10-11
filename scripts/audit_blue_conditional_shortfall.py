"""Explain failed TRAIN correction without inspecting development or test images."""
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.audit_blue_conditional_trial import OUTPUT as TRIAL

OUTPUT = Path('research/balloon-red-blue-20261008/blue-conditional-shortfall-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial = read_json(TRIAL)
    if trial['eligible_for_development'] is not False:
        raise ValueError('Expected failed training trial')
    results = {}
    for name, report in trial['trials'].items():
        run = Path(report['path'])
        for file, key in (('predictions.npz', 'predictions_sha256'), ('epochs.csv', 'history_sha256')):
            if sha256(run / file) != report[key]:
                raise ValueError('Training evidence changed')
        with np.load(run / 'predictions.npz') as arrays:
            groups = {}
            for prefix, label in (('new_', 0), ('photo_', 2), ('batch_', 2)):
                before, after = softmax(arrays[prefix+'before']), softmax(arrays[prefix+'after'])
                # For either non-red class, an unconstrained correction can approach
                # 1-p(red). This is an upper bound, NOT a learned accuracy result.
                ceiling = 1.-before[:, 1]
                groups[prefix] = dict(rows=len(before), target_class=label,
                    red_mass_blocks_acceptance=int((ceiling <= .8).sum()),
                    actual_accepted=int((after[:, label] >= .8).sum()),
                    correct_argmax=int((after.argmax(1) == label).sum()),
                    minimum_reachable_probability_ceiling=float(ceiling.min()),
                    actual_probabilities=after[:, label].tolist())
        with (run / 'epochs.csv').open() as stream:
            loss = np.array([float(r['loss']) for r in csv.DictReader(stream)])
        results[name] = dict(groups=groups,
            first40_median_loss=float(np.median(loss[:40])),
            preceding40_median_loss=float(np.median(loss[-80:-40])),
            last40_median_loss=float(np.median(loss[-40:])),
            final_loss=float(loss[-1]), frozen_parent_unchanged=report['frozen_parent_unchanged'],
            max_red_probability_delta=report['max_red_probability_delta'])
    write_json(OUTPUT, dict(trials=results, training_receipt_sha256=sha256(TRIAL),
        script_sha256=sha256(__file__),
        conclusion='Frozen red mass does not block these new blue/background TRAIN examples. Loss remains decreasing; optimization versus feature-capacity limitation is not yet resolved.',
        next_experiment='Matched fixed400epoch correction with initial learning rate0.001 instead of0.0001; unchanged data, architecture, seeds and gates.',
        scope='TRAIN diagnosis only; no development images or independent test evaluated.',
        test_evaluated=False, pi_tested=False, deployment_approved=False))
    print({name: {group: value['red_mass_blocks_acceptance'] for group, value in row['groups'].items()}
           for name, row in results.items()})


if __name__ == '__main__':
    audit()
