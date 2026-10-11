import numpy as np
import pytest

from scripts.edge_crop_localizer_data import WINDOWS, crop_scene


def target(box, label='blue_balloon', ident=1):
    return dict(box=box, label=label, annotation_id=ident)


def test_visible_partial_box_is_clipped_and_outside_box_removed():
    image = np.zeros((240, 320, 3), np.uint8)
    rows = [target([200, 100, 280, 200]), target([250, 190, 300, 220], ident=2)]
    rgb, truth, supervision, collisions = crop_scene(image, rows, WINDOWS[0])
    assert rgb.shape == image.shape and len(truth) == 1 and collisions == 0
    assert truth[0]['box'] == pytest.approx([200*4/3, 100*4/3, 320, 240])
    assert supervision[..., 8].sum() == 1
    assert rows[0]['box'] == [200, 100, 280, 200]


def test_tiny_visible_sliver_is_not_labeled_background():
    _, truth, supervision, _ = crop_scene(np.zeros((240, 320, 3), np.uint8),
        [target([239.5, 20, 260, 40])], WINDOWS[0])
    assert len(truth) == 1 and truth[0]['box'][2]-truth[0]['box'][0] < 1
    assert supervision[..., 1].max() == 1 and supervision[..., 8].sum() == 1


def test_empty_complete_scene_has_only_negative_supervision():
    _, truth, supervision, _ = crop_scene(np.zeros((240, 320, 3), np.uint8), [], WINDOWS[2])
    assert truth == [] and np.all(supervision[..., 2:4] == 1)
    assert not np.any(supervision[..., :2]) and not np.any(supervision[..., 4:])


def test_fixed_crop_contract_and_annotations_fail_closed():
    image = np.zeros((240, 320, 3), np.uint8)
    with pytest.raises(ValueError, match='fixed crop'):
        crop_scene(image, [], [0, 0, 100, 100])
    with pytest.raises(ValueError, match='annotation'):
        crop_scene(image, [target([0, 0, 321, 20])], WINDOWS[0])
