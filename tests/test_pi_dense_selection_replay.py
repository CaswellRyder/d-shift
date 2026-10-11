import ast
import inspect

import numpy as np
import pytest

from scripts.audit_balloon_dense_proposals import decode
from scripts.build_dense_selection_replay import decoder_source
from scripts.pi_dense_selection_replay import GuardedDense, process, compare


class Classifier:
    def predict(self, rgb):
        return dict(label='blue_balloon',score=.99,scores=[.005,.005,.99],accepted=True)


class Dense:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def predict(self, rgb):
        self.calls += 1
        if self.fail:
            raise RuntimeError('inference failed')
        return [dict(box=[40,40,60,60],color_group=1,proposal_score=.8)]


def proposals(nested):
    boxes = [[0,0,100,100]] + ([[40,40,60,60]] if nested else [])
    return [dict(box=b,crop_box=b,color_group=1,area=100,proposal_score=1) for b in boxes]


def test_generated_decoder_is_exact_frozen_function():
    generated = ast.parse(decoder_source()).body[-1]
    canonical = ast.parse(inspect.getsource(decode)).body[0]
    assert ast.dump(generated) == ast.dump(canonical)


def test_no_gate_no_extra_inference(monkeypatch):
    monkeypatch.setattr('scripts.pi_dense_selection_replay.experimental',lambda *a:proposals(False))
    net = Dense()
    row = process(np.zeros((240,320,3),np.uint8),Classifier(),GuardedDense(net))
    assert net.calls == row['dense_calls'] == 0
    assert row['detections'] == row['baseline']


def test_nested_pair_runs_one_localizer_and_keeps_crop_calls(monkeypatch):
    monkeypatch.setattr('scripts.pi_dense_selection_replay.experimental',lambda *a:proposals(True))
    net = Dense()
    row = process(np.zeros((240,320,3),np.uint8),Classifier(),GuardedDense(net))
    assert row['neural_calls'] == 2
    assert net.calls == row['dense_calls'] == 1
    assert not row['detections'][0]['accepted'] and row['detections'][1]['accepted']
    assert row['baseline'][0]['accepted'] and not row['baseline'][1]['accepted']


def test_first_failure_latches_baseline_until_process_restart(monkeypatch):
    monkeypatch.setattr('scripts.pi_dense_selection_replay.experimental',lambda *a:proposals(True))
    net = Dense(fail=True)
    guarded = GuardedDense(net)
    for count in (1,0):
        row = process(np.zeros((240,320,3),np.uint8),Classifier(),guarded)
        assert row['dense_calls'] == count
        assert row['dense_error'] == 'RuntimeError: inference failed'
        assert row['detections'] == row['baseline']
    assert net.calls == 1


def test_selection_reason_mismatch_is_not_hidden_by_score_tolerance(monkeypatch):
    monkeypatch.setattr('scripts.pi_dense_selection_replay.experimental',lambda *a:proposals(True))
    row = process(np.zeros((240,320,3),np.uint8),Classifier(),GuardedDense(Dense()))
    changed = [dict(r) for r in row['detections']]
    changed[0]['dense_supported_child'] = 20
    with pytest.raises(ValueError,match='reason'):
        compare(changed,row['detections'])


def test_summary_rechecks_gate_calls_and_records(monkeypatch):
    import copy
    from scripts.summarize_dense_selection_replay import validate_rows
    monkeypatch.setattr('scripts.pi_dense_selection_replay.experimental',lambda *a:proposals(True))
    row = process(np.zeros((240,320,3),np.uint8),Classifier(),GuardedDense(Dense()))
    row.update(id=0,panel='fixture')
    frames = [dict(id=0,panel='fixture',baseline=row['baseline'],dense_calls=1,
                   selected={'47':row['detections']})]
    report = dict(seed='47',frames=1,rows=[row],dense_calls=1,max_host_score_delta=0.)
    assert validate_rows(frames,report,'47')['skipped_calls'] == 0
    for key, value in (('dense_calls',0),('dense_error','failure'),('neural_calls',1),('id',2)):
        changed = copy.deepcopy(report)
        changed['rows'][0][key] = value
        with pytest.raises(ValueError):
            validate_rows(frames,changed,'47')
