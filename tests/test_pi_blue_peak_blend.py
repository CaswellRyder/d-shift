import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import pi_blue_peak_blend as runner


@pytest.mark.parametrize('policy,blend',[('original',False),('blend',True),('head_blend',True)])
def test_one_localizer_call_and_fixed_classifier_workload(monkeypatch,policy,blend):
    calls = []
    base = dict(box=[5,5,15,15],crop_box=[3,3,17,17],area=100.,color_group=1,proposal_score=.9)
    monkeypatch.setattr(runner,'search',lambda image,native,bright:[dict(base)])
    monkeypatch.setitem(sys.modules,'scripts.current_dense_frozen',SimpleNamespace(supplement=lambda a,b:a+b))

    class Network:
        def predict(self,rgb,flag):
            calls.append(('localizer',flag))
            return []

    class Classifier:
        def predict(self,crop):
            calls.append(('classifier',crop.shape))
            return dict(label='blue_balloon',accepted=True,score=.99,scores=[.005,.005,.99])

    result = runner.process(np.zeros((240,320,3),np.uint8),Classifier(),Network(),None,policy)
    assert calls == [('localizer',blend),('classifier',(14,14,3))]
    assert result['localizer_calls'] == result['neural_calls'] == 1
    assert len(result['detections']) == 1 and result['detections'][0]['accepted']
    assert np.isclose(result['processing_ms'],sum(result[k] for k in runner.TIMINGS[1:]),atol=1e-8,rtol=0)


def test_unknown_policy_fails_before_execution():
    with pytest.raises(ValueError,match='policy'):
        runner.process(None,None,None,None,'all_at_once')


def test_counterbalanced_order_has_two_runs_per_method():
    assert runner.ORDER == ('original','blend','head_blend','head_blend','blend','original')
    assert runner.ORDER == tuple(reversed(runner.ORDER))
