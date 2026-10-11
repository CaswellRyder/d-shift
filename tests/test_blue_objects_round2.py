from scripts.acquire_blue_objects_round2 import previous_ids
from scripts.acquire_blue_objects import choose, CLASSES
from scripts.blue_object_round2_data import checked_extra, QUEUE, REVIEW
from scripts.openimages_balloon_training import review_labels
from dtr.data import read_json
import pytest


def test_prior_rejected_downloads_are_not_retried_as_novel_sources():
    rows = [dict(images=[dict(image_id='a')], screened_out=[dict(image_id='b'), dict(error='no ID')]),
            dict(images=[dict(image_id='c')])]
    assert previous_ids(rows) == {'a', 'b', 'c'}
    label = next(iter(CLASSES))
    chosen = choose({i: [dict(LabelName=label)] for i in ('a', 'b', 'c', 'd')}, previous_ids(rows))
    assert chosen == {'d': label}


def test_round2_partition_excludes_other_color_candidates():
    review = read_json(REVIEW)
    labels = review_labels(review, 21)
    assert len(labels) == 12
    assert set(labels.values()) == {'background'}
    assert not set(labels).intersection(review['exclude'])


@pytest.mark.skipif(not (QUEUE / 'review.json').exists(), reason='Optional public source archive')
def test_round2_crops_reproduce_licensed_training_sources():
    rows = checked_extra()
    assert len(rows) == 12
    assert len({r['sha256'] for r in rows}) == 12
    assert all(r['attribution']['Subset'] == 'train' for r in rows)
