from collections import Counter

import pytest

from scripts import balloon_expansion4 as expansion


class Predictor:
    metadata = {'classes':expansion.CLASSES}

    def __init__(self,*args,**kwargs):
        pass

    def predict(self,path):
        return dict(scores=[.4,.3,.3])


def test_hard_selection_uses_reviewed_label_not_argmax(monkeypatch):
    class Fixed:
        def predict(self,path):
            return dict(scores=[.8,.15,.05])
    rows = [dict(id=i,label=c,path=str(i)) for i,c in enumerate(expansion.CLASSES)]
    selected, audit = expansion.hard_examples(rows,Fixed())
    assert [r['id'] for r in selected] == [1,2]
    assert not audit[0]['hard']


def test_hard_control_matches_counts_and_preserves_ordinary_rows(monkeypatch):
    ordinary = [dict(label=c,path=str(i),sha256=str(i)) for i,c in enumerate(expansion.CLASSES)]
    fragments = [dict(label='balloon_fragment',path='fragment',sha256='fragment')]
    extra = [dict(id=i,label=c,path='extra'+str(i),sha256='extra'+str(i)) for i,c in enumerate(expansion.CLASSES)]
    monkeypatch.setattr(expansion,'checked_data',lambda:(ordinary,fragments))
    monkeypatch.setattr(expansion,'checked_extra',lambda:extra)
    monkeypatch.setattr(expansion,'Predictor',Predictor)
    control,_ = expansion.data('control')
    admit,_ = expansion.data('admit')
    assert len(admit) == len(control) == 19
    assert admit[:7] == control[:7]
    assert all(r['preserve'] for r in admit[:7])
    assert all(not r['preserve'] for r in admit[7:])
    assert Counter(r['label'] for r in control) == Counter(r['label'] for r in admit)
    assert control == expansion.data('control')[0]
    with pytest.raises(ValueError,match='matched'):
        expansion.data('invalid')


def test_nonfinite_mining_scores_fail_closed():
    class Invalid:
        def predict(self,path):
            return dict(scores=[float('nan'),.5,.5])
    with pytest.raises(ValueError,match='scores'):
        expansion.hard_examples([dict(id=0,label='background',path='x')],Invalid())


def test_weighting_cannot_silently_change_other_experiments(tmp_path):
    from scripts.train_balloon_reviewed_clutter import train
    for kwargs in ({'hard_weight':16},{'expansion4':True,'hard_weight':5},
                   {'expansion4':True,'expansion':True}):
        with pytest.raises(ValueError):
            train('admit',47,tmp_path/'unused',**kwargs)
    assert not (tmp_path/'unused').exists()


def test_training_feasibility_requires_fit_and_retention():
    from scripts.evaluate_balloon_expansion4 import feasible
    fit = dict(reviewed_crop_count=5,reviewed_crop_correct=4,
               ordinary_training_rows=1487,ordinary_training_correct=1450)
    assert feasible(fit)
    assert not feasible(dict(fit,reviewed_crop_correct=3))
    assert not feasible(dict(fit,ordinary_training_correct=1400))
    assert not feasible(dict(fit,reviewed_crop_count=4))
