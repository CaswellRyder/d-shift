from scripts.mine_red_blue_negatives import safe_negative


def test_any_balloon_box_disqualifies_negative_even_with_unknown_color():
    assert safe_negative([1, 1, 5, 5], [[10, 10, 20, 20]])
    assert not safe_negative([1, 1, 5, 5], [[4, 4, 9, 9]])
    assert not safe_negative([1, 1, 5, 5], [[0, 0, 20, 20]])
