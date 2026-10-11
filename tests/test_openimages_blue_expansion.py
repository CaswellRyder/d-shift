from copy import deepcopy

import pytest

from scripts.openimages_blue_expansion import selection


def fixture():
    frames = [{'image_id': 'abc'}]
    samples = [dict(id=0, split='train', source_image='openimages/train/abc',
                    frame_id=0, box=[1, 1, 9, 9], crop_box=[0, 0, 10, 10])]
    review = dict(split='train', label='blue_balloon', deployment_approved=False,
                  selected_crop_ids=[0], manual_boxes=[dict(frame_id=0, box=[20, 20, 40, 40])])
    return review, samples, frames


def test_selection_uses_only_explicit_positive_decisions():
    review, samples, frames = fixture()
    result = selection(review, samples, frames)
    assert len(result) == 2
    assert [r['queue_id'] for r in result] == [0, None]
    assert result[1]['crop_box'][0] < 20


@pytest.mark.parametrize('ids', [[True], [0, 0], [-1], [1]])
def test_bad_crop_ids(ids):
    review, samples, frames = fixture()
    review['selected_crop_ids'] = ids
    with pytest.raises(ValueError):
        selection(review, samples, frames)


@pytest.mark.parametrize('box', [[-1, 0, 8, 8], [1, 1, 321, 20], [1, 1, 3, 3],
                               [1, 1, 9, 241], [1, 1, True, 9], [8, 8, 2, 2]])
def test_bad_manual_boxes(box):
    review, samples, frames = fixture()
    review['manual_boxes'][0]['box'] = box
    with pytest.raises(ValueError):
        selection(review, samples, frames)


def test_duplicate_manual_selection_rejected():
    review, samples, frames = fixture()
    review['manual_boxes'].append(deepcopy(review['manual_boxes'][0]))
    with pytest.raises(ValueError):
        selection(review, samples, frames)


@pytest.mark.parametrize('field,value', [('split', 'validation'), ('label', 'background'),
                                       ('deployment_approved', True)])
def test_scope_is_train_only(field, value):
    review, samples, frames = fixture()
    review[field] = value
    with pytest.raises(ValueError):
        selection(review, samples, frames)
