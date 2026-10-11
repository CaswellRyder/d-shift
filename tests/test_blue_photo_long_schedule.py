import numpy as np
import pytest
from scripts.blue_photo_long_schedule import learning_rate
from scripts.audit_blue_photo_convergence import verify_optimizer_config

def test_extended_schedule_and_serialization_are_fixed():
    import keras
    p=dict(learning_rate=.0001,learning_rate_schedule=dict(kind='cosine',decay_steps=44400,alpha=.1))
    fn=learning_rate(p)
    np.testing.assert_allclose([float(fn(i)) for i in (0,22200,44400)],[1e-4,5.5e-5,1e-5],rtol=1e-5)
    config=keras.optimizers.serialize(keras.optimizers.Adam(fn))
    verify_optimizer_config(config)
    config['config']['learning_rate']['config']['decay_steps']=26640
    with pytest.raises(ValueError):
        verify_optimizer_config(config)
    p['learning_rate_schedule']['decay_steps']=26640
    with pytest.raises(ValueError):
        learning_rate(p)
