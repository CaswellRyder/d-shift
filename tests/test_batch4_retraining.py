import math

from dtr.data import read_json
from scripts.train_batch4_retraining import PROTOCOL


def test_fixed_matched_schedule_and_same_small_architecture():
    p = read_json(PROTOCOL)
    previous = read_json('configs/background-convergence-20261011.json')
    for key in ('parent', 'parent_sha256', 'epochs', 'seeds', 'batch_size', 'learning_rate',
                'cosine_alpha', 'hard_weight', 'protection_weight', 'protection_probability_cap',
                'parent_acceptance_threshold', 'parameters'):
        assert p[key] == previous[key]
    assert p['arms'] == ['control', 'admit']
    assert p['rows'] == 2074 + 15*4
    assert p['optimizer_steps'] == math.ceil(p['rows']/p['batch_size'])*p['epochs'] == 53600
    assert p['test_evaluated'] is False and p['deployment_approved'] is False
