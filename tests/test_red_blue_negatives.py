import pytest

from scripts.mine_red_blue_negatives import mine, safe_negative


def test_any_balloon_box_disqualifies_negative_even_with_unknown_color():
    assert safe_negative([1, 1, 5, 5], [[10, 10, 20, 20]])
    assert not safe_negative([1, 1, 5, 5], [[4, 4, 9, 9]])
    assert not safe_negative([1, 1, 5, 5], [[0, 0, 20, 20]])


@pytest.mark.parametrize("variant,limit", [("invalid", 2), ("mser", 0), ("mser", 13)])
def test_mining_bounds_fail_before_opening_data(tmp_path, variant, limit):
    with pytest.raises(ValueError, match="1..12"):
        mine("missing-manifest", "missing-model", tmp_path / "out", variant, limit)
    assert not (tmp_path / "out").exists()
