import numpy as np
import pytest

from scripts.window1_repair_constraints import rebase, assemble


def test_rebase_preserves_absolute_gap_requirements_for_both_signs():
    reference = np.array([[1., 2., 3.]])
    current = np.array([[5., 1., 7.]])
    ids = [(0, 2, 0), (0, -2, 1), (0, 0, 1)]
    original = np.array([.1, .2, .3])
    shifted = rebase(original, ids, reference, current)
    for i, (_, target, rival) in enumerate(ids):
        t, r = (target, rival) if target >= 0 else (rival, -target)
        assert shifted[i]+current[0, t]-current[0, r] == pytest.approx(original[i]+reference[0, t]-reference[0, r])


def test_equal_reference_needs_no_translation():
    values = np.array([[0., 1., 2.]])
    np.testing.assert_array_equal(rebase([3.], [(0, 1, 0)], values, values), [3.])


@pytest.mark.parametrize('identity', [[(3, 1, 0)], [(0, -3, 0)], [(0, 1, 3)]])
def test_invalid_identity_rejected(identity):
    with pytest.raises(ValueError):
        rebase([1.], identity, np.zeros((1, 3)), np.zeros((1, 3)))


def test_only_two_reviewed_new_scene_proposals_exempted():
    design = np.ones((8, 2))
    logits = np.tile([4., 0., 0.], (8, 1))
    a, b, ids, objective = assemble(design, logits, logits, np.array([0, 1]), np.array([1, 2]), 1, [5, 6])
    assert a.shape == (len(b), 4)
    assert set(ids[ids[:, 0] == 2, 1]) == {4, 7}
    assert objective.tolist() == [0, 1, 4, 7]
    with pytest.raises(ValueError):
        assemble(design, logits, logits, np.array([0, 1]), np.array([1, 2]), 1, [4, 5])
