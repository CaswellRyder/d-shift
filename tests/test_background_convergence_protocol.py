from dtr.data import read_json


def test_long_schedule_preserves_data_and_loss_controls():
    short = read_json('configs/background-preservation-20261011.json')
    long = read_json('configs/background-convergence-20261011.json')
    for field in ('parent', 'parent_sha256', 'arms', 'seeds', 'batch_size', 'learning_rate',
                  'cosine_alpha', 'hard_weight', 'protection_weight', 'protection_probability_cap',
                  'parent_acceptance_threshold', 'parameters', 'rows', 'data', 'intervention'):
        assert short[field] == long[field]
    assert short['epochs'] == 80 and long['epochs'] == 400
    assert long['optimizer_steps'] == 5*short['optimizer_steps'] == 52800
    assert all(long[k] is False for k in ('development_evaluated', 'test_evaluated', 'deployment_approved'))
