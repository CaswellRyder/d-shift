import numpy as np
import pytest

from scripts.batch5_head_constraints import assemble
from scripts.balloon_head_constraints import solve
from scripts.balloon_scene_constraints import decisions


def sample():
    # old background, newly selected blue, retained red, excluded hard blue proposal
    design = np.eye(4)
    logits = np.array([[4., 0., 0.], [2., 0., 0.], [0., 4., 0.], [2., 0., 0.]])
    return design, logits


def test_correction_and_retention_solve_without_relabeling_exempt_proposal():
    design, logits = sample()
    a, b, ids, objective = assemble(design, logits, np.array([0]), 1, 1, 1, 3)
    assert not np.any(ids[:, 1] == 3)
    np.testing.assert_array_equal(objective, [0, 2])
    delta, result = solve(a, b)
    assert result['success']
    after = logits.copy()
    after[:, 1:] += design@delta.reshape(4, 2)
    np.testing.assert_array_equal(decisions(after), [0, 2, 1, 0])
    assert np.max(b-a@delta) < 1e-6


@pytest.mark.parametrize('exempt', [-1, 0, 2, 4])
def test_only_new_scene_proposal_can_be_exempt(exempt):
    design, logits = sample()
    with pytest.raises(ValueError):
        assemble(design, logits, np.array([0]), 1, 1, 1, exempt)


def test_rejected_proposals_not_forced_to_background_argmax():
    design, logits = sample()
    logits[2] = [0., .2, .1]  # Red argmax, confidence below runtime threshold.
    _, _, ids, _ = assemble(design, logits, np.array([0]), 1, 1, 1, 3)
    relevant = ids[(ids[:, 0] == 2) & (ids[:, 1] == 2)]
    assert set(relevant[:, 2]) == {-1, -2}
