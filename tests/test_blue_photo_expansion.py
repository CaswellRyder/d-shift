import numpy as np
import pytest

from scripts import blue_photo_expansion_data as data
from scripts import train_blue_photo_expansion as train


def test_new_photos_are_hard_labels_with_matched_existing_controls(monkeypatch):
    rows = [dict(label='blue_balloon', preserve=i < 1602, sha256=f'old{i}', value=i) for i in range(1750)]
    original = np.arange(1750, dtype=np.float32)[:, None]
    def base(mode):
        assert mode == 'admit'
        return rows, np.full(1750, 2), np.arange(1750) < 1602, [original, np.ones((69, 1))]
    monkeypatch.setattr(data, 'base_inputs', base)
    extra = [dict(label='blue_balloon', sha256=f'new{i}', value=2000+i) for i in range(7)]
    monkeypatch.setattr(data, 'checked_extra', lambda: extra)
    monkeypatch.setattr(data, 'load_crop', lambda r: (np.array([r['value']], np.float32), 0.))
    c, a = data.inputs('control'), data.inputs('admit')
    for result in (c, a):
        assert len(result[0]) == 1764 and result[2].sum() == 1602
        np.testing.assert_array_equal(result[3][0][:1750], original)
        assert result[0][:1750] == rows
        assert all(not r['preserve'] and r['label'] == 'blue_balloon' for r in result[0][1750:])
        assert result[0][1750:1757] == result[0][1757:]
    np.testing.assert_array_equal(c[3][-1], a[3][-1])
    assert all(r['value'] < 2000 for r in c[0][1750:])
    assert all(r['value'] >= 2000 for r in a[0][1750:])
    with pytest.raises(ValueError):
        data.inputs('test')
    extra[0]['sha256'] = 'old0'
    with pytest.raises(ValueError):
        data.inputs('admit')


def test_photo_gate_requires_all_new_positives_and_base_retention(monkeypatch):
    base_pass = True
    def base(a):
        assert len(a['after']) == 1750
        return dict(training_gate_pass=base_pass)
    monkeypatch.setattr(train, 'base_calculate', base)
    a = {k: np.zeros((1764, 3)) for k in ('before', 'after')}
    a.update(labels=np.zeros(1764), ordinary=np.ones(1764, bool), photo_after=np.tile([0., 0., 10.], (7, 1)))
    assert train.calculate(a)['training_gate_pass']
    base_pass = False
    assert not train.calculate(a)['training_gate_pass']
    base_pass = True
    a['photo_after'][0] = [10., 0., 0.]
    assert train.calculate(a)['photo_blue_accepted'] == 6
    assert not train.calculate(a)['training_gate_pass']
    a['photo_after'][0] = np.nan
    with pytest.raises(ValueError):
        train.calculate(a)
