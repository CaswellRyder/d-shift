import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'orange_supplement', Path(__file__).parents[1]/'scripts/research_orange_supplement.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def candidate(x, color=0):
    return dict(box=[x, 0, x+10, 10], color_group=color)


def test_preserves_baseline_and_caps_additions():
    baseline = [candidate(0), candidate(20, 1)]
    trial = module.supplement(baseline, [candidate(0), candidate(40, 1),
                                       candidate(60), candidate(80), candidate(100)])
    assert trial[:2] == baseline
    assert len(baseline) == 2
    assert [r['box'][0] for r in trial] == [0, 20, 60, 80]


def test_zero_budget_and_duplicates():
    assert module.supplement([], [candidate(0)], extra=0) == []
    assert len(module.supplement([], [candidate(0), candidate(0)])) == 1


@pytest.mark.parametrize('extra', [-1, 3, 1.5, True])
def test_invalid_budget(extra):
    with pytest.raises(ValueError):
        module.supplement([], [], extra=extra)
