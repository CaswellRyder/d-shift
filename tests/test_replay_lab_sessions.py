import pytest

import numpy as np

from scripts.replay_lab_sessions import SimulatedWorker, estimate_dispatch_ms, fit_cost_model


def test_latency_delivers_on_first_eligible_frame_and_resubmits_that_frame():
    worker = SimulatedWorker(lambda rgb, phase: ([dict(label='green_balloon')], 4),
                             lambda phase, crops: .13)
    assert worker.submit(None, 1.0, 'balloon')
    for now in (1.05, 1.10):
        worker.now = now
        assert worker.poll() is None
        assert not worker.submit(None, now, 'balloon')
    worker.now = 1.16
    result = worker.poll()
    assert result['frame_time'] == 1.0  # Confirmation age uses submission, not delivery.
    assert result['search_ms'] == 130
    assert worker.poll() is None
    assert worker.submit(None, 1.16, 'balloon')
    worker.now = 1.28
    assert worker.poll() is None
    worker.now = 1.30
    assert worker.poll()['frame_time'] == 1.16


def test_zero_latency_still_delivers_on_next_poll_not_in_submit():
    worker = SimulatedWorker(lambda rgb, phase: ([], 0), lambda phase, crops: 0)
    worker.now = 2.0
    assert worker.poll() is None
    assert worker.submit(None, 2.0, 'goal')
    assert not worker.submit(None, 2.0, 'goal')
    worker.now = 2.05
    assert worker.poll()['frame_time'] == 2.0


def test_delivery_at_exact_deadline():
    worker = SimulatedWorker(lambda rgb, phase: ([], 1), lambda phase, crops: .125)
    assert worker.submit(None, 1.0, 'goal')
    worker.now = 1.125
    assert worker.poll()['frame_time'] == 1.0


@pytest.mark.parametrize('latency', [-1, float('nan'), float('inf')])
def test_invalid_latency_is_rejected(latency):
    worker = SimulatedWorker(lambda rgb, phase: ([], 1), lambda phase, crops: latency)
    with pytest.raises(ValueError, match='Latency'):
        worker.submit(None, 1, 'goal')


def test_cost_fit_uses_submission_frame_and_historical_balloon_budget(monkeypatch, tmp_path):
    # fixed=200 ms, each crop=100 ms. The current balloon filter must not
    # reinterpret those OLD eight-crop timings as a new four-crop search.
    monkeypatch.setattr('scripts.replay_lab_sessions.read_rgb',
                        lambda take, row: np.array([row['frame_time'] + 1]))
    monkeypatch.setattr('scripts.replay_lab_sessions.goal_candidates',
                        lambda rgb: [None] * int(rgb[0]))
    goal = dict(path=tmp_path / 'goal', rows=[
        dict(frame_time=0, search_result=dict(frame_time=-1, phase='goal', search_ms=9999)),
        dict(frame_time=1, search_result=dict(frame_time=0, phase='goal', search_ms=300)),
        dict(frame_time=2, search_result=dict(frame_time=1, phase='goal', search_ms=400)),
    ])
    balloon = dict(path=tmp_path / 'balloon', rows=[dict(frame_time=0),
        dict(frame_time=1, search_result=dict(frame_time=0, phase='balloon', search_ms=1000))])
    fit = fit_cost_model([goal, balloon])
    assert fit['fixed_ms'] == pytest.approx(200)
    assert fit['per_crop_ms'] == pytest.approx(dict(goal=100, balloon=100))
    assert fit['excluded_unrecorded_submissions'] == 1
    assert [s['crops'] for s in fit['samples']] == [1, 2, 8]


def test_dispatch_overhead_delays_delivery_but_not_the_result_age_origin():
    worker = SimulatedWorker(lambda rgb, phase: ([], 1), lambda phase, crops: .807, dispatch_s=.022)
    assert worker.submit(None, 1.0, 'balloon')
    worker.now = 1.808  # Past search_ms alone, before the frame hand-off finishes.
    assert worker.poll() is None
    worker.now = 1.83
    result = worker.poll()
    assert result['frame_time'] == 1.0
    assert result['search_ms'] == pytest.approx(807)


@pytest.mark.parametrize('dispatch', [-0.001, float('nan')])
def test_invalid_dispatch_is_rejected(dispatch):
    with pytest.raises(ValueError, match='Dispatch'):
        SimulatedWorker(lambda rgb, phase: ([], 0), lambda phase, crops: 0, dispatch_s=dispatch)


def test_dispatch_fit_reproduces_logged_delivery_frames():
    # 50 ms frames. Result A (ready 80 ms, logged at 100) needs d <= 20; result B
    # (ready 190 ms, logged at 250, not 200) needs d > 10. Zero overhead makes B early.
    rows = [dict(frame_time=t / 1000) for t in range(0, 500, 50)]
    rows[2]['search_result'] = dict(frame_time=0.0, search_ms=80)
    rows[5]['search_result'] = dict(frame_time=0.1, search_ms=90)
    fit = estimate_dispatch_ms([dict(rows=rows)])
    assert fit['exact'] == 2 and fit['early'] == 0 and fit['late'] == 0
    assert 10 < fit['dispatch_ms'] <= 20
    assert estimate_dispatch_ms([dict(rows=rows[:1])])['dispatch_ms'] == 0
