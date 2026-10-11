import copy

import pytest

from scripts.audit_blue_photo_cosine import verify_optimizer_config


def test_saved_schedule_must_match_the_declared_experiment():
    config = dict(class_name='Adam', config={'learning_rate': dict(class_name='CosineDecay',
        config=dict(initial_learning_rate=.0001, decay_steps=26640, alpha=.1, warmup_target=None, warmup_steps=0))})
    verify_optimizer_config(config)
    for key, value in [('initial_learning_rate', .001), ('decay_steps', 240), ('alpha', 0.), ('warmup_steps', 10)]:
        changed = copy.deepcopy(config)
        changed['config']['learning_rate']['config'][key] = value
        with pytest.raises(ValueError):
            verify_optimizer_config(changed)
    config['config']['learning_rate'] = .0001
    with pytest.raises(ValueError):
        verify_optimizer_config(config)
