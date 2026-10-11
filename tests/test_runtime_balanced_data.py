from collections import Counter

import numpy as np
import pytest

from scripts import runtime_balanced_data as data


def rows(prefix,counts):
    return [dict(label=label,sha256=f'{prefix}-{label}-{i}',preserve=True,view='original')
        for label,n in counts.items() for i in range(n)]


def test_controls_are_class_matched_distinct_deterministic_and_shared_views():
    base = rows('base',dict(background=20,red_balloon=50,blue_balloon=50))
    extra = rows('new',dict(background=2,red_balloon=33,blue_balloon=29))
    control = data.choose(base,extra,'control')
    admit = data.choose(base,extra,'admit')
    assert control == data.choose(base,extra,'control')
    for result in (control,admit):
        assert len(result) == 128
        assert Counter(r['label'] for r in result) == dict(background=4,red_balloon=66,blue_balloon=58)
        assert len({r['sha256'] for r in result}) == 64
        assert all(not r['preserve'] for r in result)
        assert [r['sha256'] for r in result[:64]] == [r['sha256'] for r in result[64:]]
        assert all(r['view'] == 'original' for r in result[:64])
        assert all(r['view'] == 'horizontal_flip' for r in result[64:])
    assert not ({r['sha256'] for r in control} & {r['sha256'] for r in admit})
    assert all(r['preserve'] and r['view'] == 'original' for r in base+extra)


def test_control_shortage_and_invalid_arm_rejected():
    extra = rows('new',dict(background=2,red_balloon=33,blue_balloon=29))
    with pytest.raises(ValueError,match='Insufficient'):
        data.choose([],extra,'control')
    with pytest.raises(ValueError,match='declared'):
        data.choose([],extra,'unknown')


def test_reflection_preserves_canonical_values_and_input(monkeypatch):
    values = np.arange(64*64*3,dtype=np.float32).reshape(64,64,3)
    def load(row):
        assert row['view'] == 'original'
        return values,None
    monkeypatch.setattr(data,'load_crop',load)
    np.testing.assert_array_equal(data.crop(dict(view='horizontal_flip')),values[:,::-1])
    np.testing.assert_array_equal(data.crop(dict(view='original')),values)
    with pytest.raises(ValueError,match='Undeclared'):
        data.crop(dict(view='pixels_seed47'))


@pytest.mark.parametrize('old_gate,weak_mirror,expected',[(True,False,True),(False,False,False),(True,True,False)])
def test_new_gate_keeps_old_protection_and_requires_both_views(monkeypatch,old_gate,weak_mirror,expected):
    from scripts import train_runtime_balanced_trial as train
    monkeypatch.setattr(train,'prior_calculate',lambda a:dict(training_gate_pass=old_gate))
    labels = np.repeat([0,1,2],[2,33,29])
    logits = np.eye(3,dtype=np.float32)[labels]*10
    mirror = logits.copy()
    if weak_mirror:
        mirror[-1] = [0,0,1]  # Still blue argmax, but below .8.
    result = train.calculate(dict(runtime_labels=labels,crop_original_after=logits,crop_mirror_after=mirror))
    assert result['training_gate_pass'] is expected
    assert result['runtime_crop_original_passed'] == 64
    assert result['runtime_crop_mirror_passed'] == 64-int(weak_mirror)
