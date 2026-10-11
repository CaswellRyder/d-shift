import numpy as np
import pytest

from scripts.balloon_convergence import snapshot
from scripts.train_balloon_reviewed_clutter import train


def test_trace_reports_only_preserved_rows_and_reviewed_probabilities():
    logits = np.array([[3.,0,0],[0,3,0],[0,0,3]])
    result = snapshot(10,logits,[0,1,0],np.array([True,True,False]),logits[:2],[0,2])
    assert result['ordinary_rows']==result['ordinary_correct']==2
    assert result['hard_correct']==1 and result['hard_rows']==2
    assert result['reviewed_class_probabilities'][0]>.9
    assert result['reviewed_class_probabilities'][1]<.1


def test_trace_rejects_nonfinite_and_empty_preservation():
    for logits,mask in (([[float('nan'),0,1]],np.array([True])),([[1,0,0]],np.array([False]))):
        with pytest.raises(ValueError):
            snapshot(1,logits,[0],mask,[[1,0,0]],[0])


def test_long_run_is_explicit_and_does_not_change_older_experiments(tmp_path):
    for kwargs in ({'epochs':80},{'epochs':81},{'expansion4':True,'epochs':80}):
        with pytest.raises(ValueError):
            train('admit',47,tmp_path/'unused',**kwargs)
    assert not (tmp_path/'unused').exists()


def test_trajectory_verifier_rejects_missing_epochs_and_invented_fit():
    import copy
    from scripts.evaluate_balloon_convergence import check_trace
    trace = dict(records=[dict(epoch=e,ordinary_rows=1487,ordinary_correct=1450,
        hard_rows=5,hard_correct=5,hard_probabilities=[[1.,0,0]]*5,
        reviewed_class_probabilities=[1.]*5) for e in (0,1,10,20,30,40,50,60,70,80)])
    fit = dict(reviewed_crop_correct=5,ordinary_training_correct=1450)
    labels = np.zeros(5,dtype=int)
    check_trace(trace,fit,labels)
    with pytest.raises(ValueError,match='Incomplete'):
        check_trace(dict(records=trace['records'][:-1]),fit,labels)
    changed = copy.deepcopy(trace)
    changed['records'][2]['hard_correct'] = 4
    with pytest.raises(ValueError,match='trajectory'):
        check_trace(changed,fit,labels)
    with pytest.raises(ValueError,match='disagree'):
        check_trace(trace,dict(fit,reviewed_crop_correct=4),labels)


def test_development_stays_closed_when_training_feasibility_fails(tmp_path):
    from dtr.data import write_json
    from scripts.evaluate_balloon_convergence import development
    path = tmp_path/'diagnostic.json'
    write_json(path,dict(both_admit_seeds_training_feasible=False))
    with pytest.raises(ValueError,match='Both seeds'):
        development(path,tmp_path/'output.json')
    assert not (tmp_path/'output.json').exists()
