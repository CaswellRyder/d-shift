import numpy as np
import pytest
from scripts.blue_photo_schedule import learning_rate

def test_fixed_schedule_is_monotone_and_clamped():
    p=dict(learning_rate=.0001,learning_rate_schedule=dict(kind='cosine',decay_steps=26640,alpha=.1))
    fn=learning_rate(p)
    values=np.array([float(fn(s)) for s in [0,6660,13320,19980,26640,30000]])
    np.testing.assert_allclose(values[[0,2,4,5]],[1e-4,5.5e-5,1e-5,1e-5],rtol=1e-5)
    assert np.all(np.diff(values)<=0)
    p['learning_rate_schedule']['decay_steps']=240
    with pytest.raises(ValueError):
        learning_rate(p)
