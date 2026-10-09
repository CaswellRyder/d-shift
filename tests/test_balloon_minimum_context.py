import copy

import pytest

from scripts.research_balloon_context import minimum_context


@pytest.mark.parametrize("crop", [[100, 100, 104, 106], [0, 0, 5, 4], [311, 235, 320, 240]])
def test_tiny_crop_gets_context_but_detection_box_is_preserved(crop):
    source = dict(box=crop.copy(), crop_box=crop.copy(), color_group=1, proposal_score=1.0)
    before = copy.deepcopy(source)
    result = minimum_context([source], (240, 320, 3))[0]
    a, b, c, d = result["crop_box"]
    assert c - a == d - b == 16
    assert 0 <= a <= crop[0] < crop[2] <= c <= 320
    assert 0 <= b <= crop[1] < crop[3] <= d <= 240
    assert source == before and result["box"] == source["box"]


def test_larger_crops_are_unchanged():
    for crop in ([10, 10, 26, 26], [0, 0, 100, 4], [0, 0, 320, 240]):
        source = dict(box=crop, crop_box=crop)
        assert minimum_context([source], (240, 320, 3)) == [source]


def test_invalid_crop_and_image_rejected():
    for crop in ([-1, 0, 10, 10], [0, 0, 0, 5], [0.0, 0, 5, 5], [0, 0, 321, 10]):
        with pytest.raises(ValueError):
            minimum_context([dict(crop_box=crop)], (240, 320, 3))
    with pytest.raises(ValueError):
        minimum_context([], (8, 8, 3))
