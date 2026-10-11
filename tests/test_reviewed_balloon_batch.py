from copy import deepcopy

import pytest

from scripts.reviewed_balloon_batch import inventory, load_batch


def fixture():
    return ([dict(sha256=str(i), source_image=str(i//2), label='blue_balloon') for i in range(4)],
        dict(admitted_crops=4, admitted_photographs=2, label_counts={'blue_balloon': 4}, max_crops_per_photo=2))


def test_inventory_preserves_unique_views_and_source_balance():
    inventory(*fixture())


@pytest.mark.parametrize('key,value', [('admitted_crops', 5), ('admitted_photographs', 3),
                                     ('max_crops_per_photo', 1), ('label_counts', {'red_balloon': 4})])
def test_inventory_rejects_changed_admission(key, value):
    rows, review = fixture()
    review[key] = value
    with pytest.raises(ValueError):
        inventory(rows, review)


def test_duplicate_crop_is_not_extra_training_evidence():
    rows, review = fixture()
    rows[1] = deepcopy(rows[0])
    with pytest.raises(ValueError):
        inventory(rows, review)


def test_changed_review_rejected_before_data_access(tmp_path):
    path = tmp_path/'review.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='Sealed review'):
        load_batch(path, '0'*64)
