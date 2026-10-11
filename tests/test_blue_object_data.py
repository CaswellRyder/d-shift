import numpy as np
import pytest

from scripts import blue_object_data as data


def test_matched_arms_change_only_added_background_rows(monkeypatch):
    rows=[dict(label='background',preserve=True,sha256=f'old{i}',view='original',value=i) for i in range(20)]
    extra=[dict(label='background',preserve=False,sha256=f'new{i}',view='original',value=100+i) for i in range(13)]
    base=np.arange(20,dtype=np.float32)[:,None]
    monkeypatch.setattr(data,'base_inputs',lambda mode:(rows,np.zeros(20,int),np.ones(20,bool),[base,np.ones((2,1))]))
    monkeypatch.setattr(data,'checked_extra',lambda:extra)
    monkeypatch.setattr(data,'load_crop',lambda r:(np.array([r['value']],np.float32),0.))
    control,admit=data.inputs('control'),data.inputs('admit')
    assert control[0][:20]==admit[0][:20]
    assert len(control[0])==len(admit[0])==46
    for result in (control,admit):
        np.testing.assert_array_equal(result[3][0][:20],base)
        assert not result[2][20:].any()
        assert len({r['sha256'] for r in result[0][20:33]})==13
        np.testing.assert_array_equal(result[3][-1],np.arange(100,113,dtype=np.float32)[:,None])
        assert result[0][20:33]==result[0][33:]
    assert all(r['value']<100 for r in control[0][20:])
    assert all(r['value']>=100 for r in admit[0][20:])
    with pytest.raises(ValueError):
        data.inputs('test')


def test_rejects_non_novel_extra_pixels(monkeypatch):
    old=dict(label='background',preserve=True,sha256='same')
    monkeypatch.setattr(data,'base_inputs',lambda mode:([old],np.array([0]),np.array([True]),[]))
    monkeypatch.setattr(data,'checked_extra',lambda:[dict(old,preserve=False)])
    with pytest.raises(ValueError,match='novel'):
        data.inputs('admit')
