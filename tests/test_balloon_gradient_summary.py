import numpy as np
import pytest

from scripts.balloon_gradient_summary import summarize


def test_opposing_directions_cancel_and_raise_other_loss():
    s=summarize({'old':np.array([1.,0.]),'scene':np.array([-1.,0.])},{'old':1,'scene':1})
    assert s['pairs']['old/scene']['cosine']==-1
    assert s['pairs']['old/scene']['directional_derivative']==1
    assert s['cancellation_ratio']==0


def test_group_contributions_respect_counts_and_zero_gradients():
    s=summarize({'a':np.array([1.,0.]),'b':np.zeros(2)},{'a':3,'b':1})
    assert s['contribution_norms']['a']==.75
    assert s['total_gradient_norm']==.75
    assert s['pairs']['a/b']['cosine'] is None
    assert s['cancellation_ratio']==1


def test_invalid_gradient_receipts_fail_closed():
    with pytest.raises(ValueError):
        summarize({'a':np.array([np.nan])},{'a':1})
    with pytest.raises(ValueError):
        summarize({'a':np.ones(1)},{'a':0})
