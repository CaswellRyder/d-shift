from scripts.prepare_indoor5_runtime_review import needs_review
from scripts.indoor5_runtime_data import validate_labels, checked_extra, REVIEW, QUEUE
from dtr.data import read_json
import pytest


def test_unmatched_accepted_proposal_goes_to_review_not_background_label():
    candidate = dict(crop_box=[0, 0, 10, 10])
    assert needs_review(dict(accepted=True, label='blue_balloon'), candidate, [])
    assert 'label' not in candidate


def test_wrong_class_proposals_on_known_positive_still_get_reviewed():
    candidate = dict(crop_box=[0, 0, 10, 10])
    assert needs_review(dict(accepted=False, label='background'), candidate, [candidate])
    assert not needs_review(dict(accepted=False, label='background'), candidate, [])


def test_review_partition_never_auto_labels_exclusions():
    review = read_json(REVIEW)
    labels = validate_labels(review, 94)
    assert len(labels) == 64
    assert not set(labels).intersection(review['exclude'])
    review['labels_by_class']['background'].append(0)
    with pytest.raises(ValueError):
        validate_labels(review, 94)


@pytest.mark.skipif(not (QUEUE / 'review.json').exists(), reason='Optional local source pixels')
def test_reviewed_runtime_pixels_reconstruct_training_sources():
    rows = checked_extra()
    assert len(rows) == 64
    assert len({r['sha256'] for r in rows}) == 64
    assert {r['id'] for r in rows if r['label'] == 'background'} == {92, 93}
    assert all(r['source_image'].startswith('train/frame_') for r in rows)
