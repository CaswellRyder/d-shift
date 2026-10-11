from copy import deepcopy

import pytest

from scripts.blue_balloon_batch3_data import EXPECTED, checked_labels


def review():
    return dict(training_approved=True, deployment_approved=False, test_evaluated=False, split='train',
        labels_by_class={label: [i for i, v in EXPECTED.items() if v == label]
                         for label in ('background', 'red_balloon', 'blue_balloon')},
        exclude=sorted(set(range(28))-set(EXPECTED)))


def test_only_explicit_crop_labels_are_approved():
    assert checked_labels(review(), 28) == EXPECTED
    assert 6 not in checked_labels(review(), 28)


@pytest.mark.parametrize('change', ['unreviewed', 'wrong_color', 'test', 'duplicate', 'count'])
def test_review_drift_fails_closed(change):
    r = deepcopy(review())
    count = 28
    if change == 'unreviewed':
        r['exclude'].remove(6)
        r['labels_by_class']['blue_balloon'].append(6)
    elif change == 'wrong_color':
        r['labels_by_class']['red_balloon'] = []
        r['labels_by_class']['blue_balloon'].append(8)
    elif change == 'test':
        r['split'] = 'test'
    elif change == 'duplicate':
        r['exclude'].append(6)
    else:
        count = 29
    with pytest.raises(ValueError):
        checked_labels(r, count)
