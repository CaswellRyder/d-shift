from collections import Counter
from pathlib import Path

import pytest

from scripts import indoor_expansion5_data as data
from dtr.data import read_json


def test_review_covers_every_crop_without_relabeling_fragments():
    review = read_json(data.REVIEW)
    assert sorted(review['admit'] + review['exclude']) == list(range(95))
    assert not set(review['admit']).intersection(review['exclude'])
    categories = [i for values in review['excluded_categories'].values() for i in values]
    assert sorted(categories) == review['exclude']
    assert len(review['reviewed_sheets']) == 6
    assert review['test_evaluated'] is False and review['deployment_approved'] is False


@pytest.mark.skipif(not data.ARCHIVE.exists(), reason='Optional pinned public training archive absent')
def test_actual_admissions_reproduce_review_and_fail_on_archive_change(monkeypatch):
    rows = data.checked_extra()
    assert len(rows) == 68
    assert Counter(r['label'] for r in rows) == {'red_balloon': 32, 'blue_balloon': 25, 'background': 11}
    assert len({r['source_image'] for r in rows}) == 30
    assert all(not Path(r['source_image']).name.lower().startswith('img_') for r in rows)
    original = data.sha256
    monkeypatch.setattr(data, 'sha256', lambda p: 'changed' if Path(p) == data.ARCHIVE else original(p))
    with pytest.raises(ValueError, match='Quarantine provenance'):
        data.checked_extra()
