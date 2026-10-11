import numpy as np
import pytest

from scripts.research_blue_component_box import refine


def row(box=None):
    return dict(box=box or [10, 10, 40, 40], crop_box=[5, 5, 45, 45],
        label='blue_balloon', score=.95, scores=[.03, .02, .95], accepted=True)


def test_refines_learned_geometry_without_changing_crop_or_classification():
    image = np.zeros((240, 320, 3), np.uint8)
    image[17:32, 20:35, 2] = 255
    before = row()
    after = refine(image, [before], 0)[0]
    assert after['box'] == [20, 17, 35, 32]
    assert after['box_refined']
    for key in ('crop_box', 'label', 'score', 'scores', 'accepted'):
        assert after[key] == before[key]
    assert before['box'] == [10, 10, 40, 40]


def test_missing_color_and_clipped_support_fall_back_not_reject():
    image = np.zeros((240, 320, 3), np.uint8)
    result = refine(image, [row()], 0)[0]
    assert result['accepted'] and result['box'] == row()['box'] and not result['box_refined']
    image[10:25, 10:25, 2] = 255
    result = refine(image, [row()], 0)[0]
    assert result['box_refinement_reason'] == 'support_touches_roi_boundary'
    assert result['box'] == row()['box'] and result['accepted']


def test_preserves_baseline_and_other_labels():
    image = np.zeros((240, 320, 3), np.uint8)
    image[17:32, 20:35, 2] = 255
    original = [row(), dict(row(), label='red_balloon'), dict(row(), accepted=False)]
    assert refine(image, original, 1) == original


def test_fragmented_mask_and_invalid_prefix():
    image = np.zeros((240, 320, 3), np.uint8)
    for y, x in ((13, 13), (20, 20), (28, 28)):
        image[y:y+4, x:x+4, 2] = 255
    result = refine(image, [row()], 0)[0]
    assert not result['box_refined']
    with pytest.raises(ValueError):
        refine(image, [row()], 2)
