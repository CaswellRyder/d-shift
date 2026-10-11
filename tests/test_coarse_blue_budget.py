import pytest

from scripts.audit_coarse_blue_budget import supplement_budget
from scripts.audit_balloon_dense_proposals import supplement


def row(i, group=1):
    return dict(box=[i * 15, 0, i * 15 + 10, 10], color_group=group)


def test_two_budget_is_original_and_expansion_preserves_base():
    base = [row(0)]
    learned = [row(0), row(1, 0)] + [row(i) for i in range(2, 12)]
    assert supplement_budget(base, learned, 2) == supplement(base, learned)
    for n in (2, 4, 6):
        result = supplement_budget(base, learned, n)
        assert result[:1] == base
        assert len(result) == n + 1
        assert all(r['color_group'] == 1 for r in result[1:])


def test_global_limit_and_undeclared_budgets():
    assert len(supplement_budget([row(i) for i in range(11)], [row(12), row(13)], 6)) == 12
    with pytest.raises(ValueError):
        supplement_budget([], [], 8)
    with pytest.raises(ValueError):
        supplement_budget([row(i) for i in range(13)], [], 2)
