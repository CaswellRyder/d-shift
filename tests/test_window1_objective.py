import numpy as np
import pytest

from scripts.diagnose_window1_objective import gradient_stats


def test_opposing_group_and_margin_direction_are_explicit():
    result = gradient_stats({'new_positive': [1., 0.], 'other': [-2., 0.]}, [np.array([-1., 0.])])
    assert result['other']['cosine_new'] == -1
    assert result['new_positive']['negative_gradient_margin_derivatives'] == [1.]
    assert result['total']['negative_gradient_margin_derivatives'] == [-1.]


def test_zero_gradient_has_no_defined_cosine():
    result = gradient_stats({'new_positive': [0., 0.]}, [np.ones(2)])
    assert result['new_positive']['cosine_new'] is None
    assert result['total']['norm'] == 0


@pytest.mark.parametrize('groups', [{}, {'new_positive': [1.], 'other': [1., 2.]},
                                  {'new_positive': [float('nan')]}])
def test_invalid_gradients_fail_closed(groups):
    with pytest.raises(ValueError):
        gradient_stats(groups, [])
