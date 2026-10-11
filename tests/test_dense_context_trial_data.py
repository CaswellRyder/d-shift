import pytest
from scripts import dense_context_trial_data as data
def test_both_arms_use_identical_admitted_curriculum(monkeypatch):
    sentinel=object()
    calls=[]
    def inputs(mode):
        calls.append(mode)
        return sentinel
    monkeypatch.setattr(data,'photo_inputs',inputs)
    assert data.inputs('control') is sentinel and data.inputs('admit') is sentinel
    assert calls==['admit','admit']
    assert data.parameters('control')==7763 and data.parameters('admit')==15667
    for fn in (data.inputs,data.parameters):
        with pytest.raises(ValueError):
            fn('test')
