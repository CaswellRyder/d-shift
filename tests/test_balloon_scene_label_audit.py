import pytest

from scripts.balloon_scene_label_audit import classify_geometry


def target(box, label='blue_balloon'):
    return dict(box=box, label=label)


def test_fragment_is_not_background():
    result = classify_geometry([0, 0, 2, 2], [0, 0, 3, 3], [target([0, 0, 20, 20])])
    assert result['category'] == 'partial_or_mixed'
    assert result['matching_truth'] == []


def test_empty_requires_no_crop_overlap():
    truth = [target([10, 10, 20, 20])]
    assert classify_geometry([0, 0, 5, 5], [0, 0, 10, 10], truth)['category'] == 'empty_crop'
    assert classify_geometry([0, 0, 5, 5], [0, 0, 12, 12], truth)['category'] == 'partial_or_mixed'


def test_mixed_context_is_not_a_single_color_label():
    truth = [target([10, 10, 20, 20]), target([22, 10, 32, 20], 'red_balloon')]
    assert classify_geometry([10, 10, 20, 20], [10, 10, 20, 20], truth)['category'] == 'matched_blue_balloon'
    assert classify_geometry([10, 10, 20, 20], [8, 8, 34, 22], truth)['category'] == 'partial_or_mixed'


@pytest.mark.parametrize('box', [[0, 0, 0, 1], [0, 0, 321, 240], [0, 0, float('nan'), 4]])
def test_invalid_geometry_fails(box):
    with pytest.raises(ValueError):
        classify_geometry(box, box, [])


def test_box_outside_crop_fails():
    with pytest.raises(ValueError):
        classify_geometry([0, 0, 10, 10], [1, 1, 9, 9], [])


def test_loss_association_requires_actual_parent_support():
    from scripts.summarize_balloon_scene_labels import loss_context
    lost = dict(source='train/example', target=target([10, 10, 20, 20]))
    row = dict(source='train/example', selected=True, teacher_decision='blue_balloon',
               box=[10, 10, 20, 20], original=dict(category='matched_blue_balloon'),
               wider=dict(category='partial_or_mixed'))
    assert loss_context(lost, [row])['all_supports_newly_mixed']
    row['selected'] = False
    with pytest.raises(ValueError):
        loss_context(lost, [row])
