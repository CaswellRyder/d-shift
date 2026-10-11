from scripts.acquire_blue_objects_round3 import previous_ids
from scripts.acquire_blue_objects import choose, CLASSES
from scripts.blue_object_round3_data import checked_extra, QUEUE, REVIEW, REVIEW_SHA
from scripts.openimages_balloon_training import review_labels
from dtr.data import read_json, sha256
import pytest


def test_round3_excludes_previously_retained_and_screened_out_ids():
    old = previous_ids([dict(images=[dict(image_id='a')], screened_out=[dict(image_id='b')])])
    label = next(iter(CLASSES))
    assert choose({i: [dict(LabelName=label)] for i in ('a', 'b', 'c')}, old) == {'c': label}


def test_round3_sealed_review_covers_exactly_thirteen_backgrounds():
    assert sha256(REVIEW) == REVIEW_SHA
    assert review_labels(read_json(REVIEW), 13) == {i: 'background' for i in range(13)}


@pytest.mark.skipif(not (QUEUE/'review.json').exists(), reason='Optional public source archive')
def test_round3_reconstructs_licensed_reviewed_pixels():
    rows = checked_extra()
    assert len(rows) == len({r['sha256'] for r in rows}) == 13
    assert len({r['source_image'] for r in rows}) == 10
    assert all(r['attribution']['Subset'] == 'train' for r in rows)
