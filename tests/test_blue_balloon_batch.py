import pytest

from dtr.data import write_json, sha256
from scripts.acquire_blue_balloon_batch import acquire, prior_sources


def test_excludes_failed_and_screened_prior_attempts(tmp_path):
    root = tmp_path / 'openimages-fixture'
    root.mkdir()
    write_json(root / 'selection.json', {'selected_image_ids': ['kept', 'failed', 'screened']})
    write_json(root / 'acquisition.json', {
        'selection_sha256': sha256(root / 'selection.json'), 'images': [{'image_id': 'kept'}]})
    ids, receipts = prior_sources(tmp_path)
    assert ids == {'kept', 'failed', 'screened'}
    assert receipts[str(root / 'acquisition.json')] == sha256(root / 'acquisition.json')
    write_json(root / 'selection.json', {'selected_image_ids': []})
    with pytest.raises(ValueError, match='identity'):
        prior_sources(tmp_path)


@pytest.mark.parametrize('size,limit', [(1, 96), (257 << 20, 96), (128 << 20, 97), (128 << 20, 0)])
def test_budget_rejected_before_network(tmp_path, size, limit):
    with pytest.raises(ValueError):
        acquire(tmp_path / 'new', size, limit)
    assert not (tmp_path / 'new').exists()
