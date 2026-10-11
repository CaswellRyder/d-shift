import numpy as np
import pytest

from scripts import blue_view_recovery_data as data


def test_recovery_arms_preserve_blue_object_curriculum(monkeypatch):
    rows=[dict(label='blue_balloon',preserve=i<1533,sha256=f'old{i}',view='original',value=i) for i in range(1681)]
    base=np.arange(1681,dtype=np.float32)[:,None]
    labels=np.full(1681,2)
    ordinary=np.arange(1681)<1533
    extra=[dict(label='blue_balloon',sha256=f'new{i}',value=2000+i) for i in range(69)]
    extra += [dict(label='red_balloon',sha256='red',value=9999)]
    monkeypatch.setattr(data,'base_inputs',lambda mode:(rows,labels,ordinary,[base,np.ones((13,1))]))
    monkeypatch.setattr(data,'checked_extra',lambda known:extra)
    monkeypatch.setattr(data,'load_crop',lambda r:(np.array([r['value']],np.float32),0.))
    c,a=data.inputs('control'),data.inputs('admit')
    for r in (c,a):
        assert len(r[0])==1750 and r[2].sum()==1602
        assert r[0][:1681]==rows
        np.testing.assert_array_equal(r[3][0][:1681],base)
        assert all(x['preserve'] and x['label']=='blue_balloon' for x in r[0][1681:])
        assert len({x['sha256'] for x in r[0][1681:]})==69
    assert all(r['value']<2000 for r in c[0][1681:])
    assert all(r['value']>=2000 for r in a[0][1681:])
    np.testing.assert_array_equal(c[3][-1],a[3][-1])
    with pytest.raises(ValueError):
        data.inputs('test')
