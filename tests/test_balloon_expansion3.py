import pytest
from collections import Counter

from scripts.balloon_expansion3 import partition


def test_review_partition_is_explicit_and_training_only():
    review = dict(admit=[0,2],exclude=[1],training_approved=True,test_evaluated=False,deployment_approved=False)
    partition(review,3)
    for change in (dict(admit=[0]),dict(admit=[0,0,2]),dict(admit=[True,2]),
                   dict(exclude=[]),dict(admit=[0,1,2]),dict(training_approved=False),
                   dict(deployment_approved=True),dict(test_evaluated=True)):
        with pytest.raises(ValueError):
            partition(dict(review,**change),3)


def test_expansion_control_matches_labels_and_preserves_ordinary_rows(monkeypatch):
    from scripts import balloon_expansion3 as expansion
    ordinary = [dict(label=label,path=f'{i}.png',sha256=str(i))
                for i,label in enumerate(('background','red_balloon','blue_balloon'))]
    fragments = [dict(label='balloon_fragment',path='fragment.png',sha256='fragment')]
    extra = [dict(label=label,path=f'extra-{i}.png',sha256=f'extra-{i}')
             for i,label in enumerate(('red_balloon','blue_balloon','background'))]
    monkeypatch.setattr(expansion,'checked_data',lambda:(ordinary,fragments))
    monkeypatch.setattr(expansion,'checked_extra',lambda:extra)
    control,_ = expansion.data('control')
    admit,_ = expansion.data('admit')
    assert len(control)==len(admit)==13
    assert control[:7]==admit[:7]
    assert all(r['preserve'] for r in admit[:7])
    assert all(not r['preserve'] for r in admit[7:])
    assert Counter(r['label'] for r in control)==Counter(r['label'] for r in admit)
    assert [r['label'] for r in control[7:]]==[r['label'] for r in admit[7:]]
    assert control==expansion.data('control')[0]
    with pytest.raises(ValueError,match='matched'):
        expansion.data('unknown')
