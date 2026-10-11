import numpy as np
import pytest

from scripts import train_blue_view_recovery as trial


def test_all_runtime_views_required_without_redefining_base(monkeypatch):
    def base(a):
        assert len(a['after'])==1681
        return dict(training_gate_pass=True)
    monkeypatch.setattr(trial,'base_calculate',base)
    arrays={k:np.zeros((1750,3)) for k in ('before','after')}
    arrays.update(labels=np.zeros(1750),ordinary=np.ones(1750,bool),
                  runtime_after=np.tile([0.,0.,10.],(69,1)))
    assert trial.calculate(arrays)['training_gate_pass']
    arrays['runtime_after'][0]=[10.,0.,0.]
    r=trial.calculate(arrays)
    assert r['runtime_blue_accepted']==68 and not r['training_gate_pass']
    arrays['runtime_after'][0]=np.nan
    with pytest.raises(ValueError):
        trial.calculate(arrays)
