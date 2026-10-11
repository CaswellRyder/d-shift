from copy import deepcopy

import numpy as np
import pytest

from scripts.prepare_current_color_clutter import select
from scripts.train_current_color_clutter import calculate


def arrays():
    labels = np.arange(1685)%3
    correct = np.eye(3,dtype=np.float32)[labels]*10
    a = dict(labels=labels,ordinary=np.arange(1685)<1533,before=correct.copy(),after=correct.copy())
    for key,n,color in (('blue_after',23,2),('clutter_after',22,0),('real_after',39,0),('new_after',15,0)):
        a[key] = np.tile(np.eye(3,dtype=np.float32)[color]*10,(n,1))
    return a


def test_gate_protects_both_colors_and_old_negative_learning():
    a = arrays()
    assert calculate(a)['training_gate_pass']
    for index in (1,2):
        b = deepcopy(a)
        b['after'][index] = [10,0,0]
        assert not calculate(b)['training_gate_pass']
    for key in ('real_after','new_after','clutter_after','blue_after'):
        b = deepcopy(a)
        b[key][0] = [0,10,0]
        assert not calculate(b)['training_gate_pass']


def test_gate_rejects_nonfinite_predictions():
    a = arrays()
    a['new_after'][0,0] = np.nan
    with pytest.raises(ValueError):
        calculate(a)


def test_review_selection_caps_sources_and_colors():
    rows = [dict(source=f'source{i//3}',crop_box=[i,0,i+1,1],
                 mining_prediction=dict(label='red_balloon',score=.99)) for i in range(60)]
    chosen = select(rows)
    assert len(chosen)==24
    assert max(sum(r['source']==s for r in chosen) for s in {r['source'] for r in chosen})==2
