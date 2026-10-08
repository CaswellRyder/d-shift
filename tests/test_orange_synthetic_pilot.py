import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('synthetic_pilot', scripts/'train_orange_synthetic_pilot.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def manifest(tmp_path):
    crop = tmp_path/'crop.png'
    crop.write_bytes(b'fixture')
    row = dict(path='crop.png', split='train', synthetic=True, label='background',
               sha256=hashlib.sha256(b'fixture').hexdigest())
    doc = dict(synthetic=True, split='train', classes=['background', 'orange_circle'], samples=[row])
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(doc))
    return path, doc


def test_seed_accepts_only_verified_training_inputs(module, tmp_path):
    path, doc = manifest(tmp_path)
    assert module.check_seed(path, doc['classes']) == doc
    (tmp_path/'crop.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='crop'):
        module.check_seed(path, doc['classes'])


@pytest.mark.parametrize('field,value', [('split', 'val'), ('synthetic', False),
                                       ('label', 'orange_balloon'), ('path', '../escape.png')])
def test_rejects_unapproved_seed_rows(module, tmp_path, field, value):
    path, doc = manifest(tmp_path)
    doc['samples'][0][field] = value
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        module.check_seed(path, doc['classes'])
