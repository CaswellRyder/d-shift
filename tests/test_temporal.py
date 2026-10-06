import json

import cv2
import numpy as np
import pytest

from dtr.temporal import TemporalVision


@pytest.mark.parametrize("shape,threshold", [((12,12,3),20), ((24,32,3),25)])
def test_native_l1_retains_integer_signature_threshold_decisions(shape,threshold):
    old = TemporalVision(None,difference_backend="numpy")
    new = TemporalVision(None,difference_backend="native")
    rng = np.random.default_rng(17)
    for _ in range(100):
        a = rng.integers(0,256,shape).astype(np.float32)
        b = rng.integers(0,256,shape).astype(np.float32)
        assert old.difference_exceeds(a,b,threshold) == new.difference_exceeds(a,b,threshold)
    a = np.zeros(shape,np.float32)
    b = np.full(shape,threshold,np.float32)
    for delta in (-1,0,1):
        b.flat[0] = threshold+delta
        assert old.difference_exceeds(a,b,threshold) == new.difference_exceeds(a,b,threshold) == (delta>0)


def test_native_l1_keeps_complete_tracking_outputs(scene,monkeypatch):
    assert TemporalVision(None).difference_backend == "native"
    monkeypatch.setattr("dtr.temporal.time.perf_counter",lambda:0.)
    old = TemporalVision(Predictor(),difference_backend="numpy")
    new = TemporalVision(Predictor(),difference_backend="native")
    for index in range(30):
        frame = np.zeros_like(scene) if index in (8,9) else np.roll(scene,index%6,axis=1)
        assert old.observe(frame,index*.1) == new.observe(frame,index*.1)
    with pytest.raises(ValueError,match="difference backend"):
        TemporalVision(None,difference_backend="bad")


class Predictor:
    metadata = {"task": "goal"}

    def __init__(self):
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        return dict(label="yellow_square",score=.95,accepted=True)


@pytest.fixture
def scene(monkeypatch):
    frame = np.zeros((80,120,3),np.uint8)
    frame[20:40,20:40] = (180,180,0)

    def found(rgb, *args, **kwargs):
        mask = np.where(rgb[:,:,0]>100,255,0).astype(np.uint8)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rows = []
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            rows.append(dict(box=[x,y,x+w,y+h],crop_box=[x,y,x+w,y+h],color_group=1))
        return rows
    monkeypatch.setattr("dtr.temporal.proposals",found)
    return frame


def test_reuses_label_and_tracks_slow_translation(scene):
    model = Predictor()
    tracker = TemporalVision(model)
    first = tracker.observe(scene,0)
    moved = np.roll(scene,2,axis=1)
    second = tracker.observe(moved,.1)
    assert model.calls == 1
    assert first["observations"][0]["classification_fresh"]
    assert not second["observations"][0]["classification_fresh"]
    assert second["observations"][0]["box"] == [22,20,42,40]
    assert second["observations"][0]["classification_age_ms"] >= 100
    json.dumps(second,allow_nan=False)


def test_disappearance_immediately_removes_cached_detection(scene):
    tracker = TemporalVision(Predictor())
    tracker.observe(scene,0)
    result = tracker.observe(np.zeros_like(scene),.1)
    assert result["target_lost"] and not result["observations"]


def test_frame_gap_clears_identity(scene):
    model = Predictor()
    tracker = TemporalVision(model)
    first = tracker.observe(scene,0)
    second = tracker.observe(scene,2)
    assert model.calls == 2
    assert first["observations"][0]["track_id"] != second["observations"][0]["track_id"]


def test_budget_and_queue_fairness(scene):
    scene[20:40,50:70] = (180,180,0)
    scene[20:40,80:100] = (180,180,0)
    model = Predictor()
    tracker = TemporalVision(model,budget=1)
    for index in range(3):
        result = tracker.observe(scene,index*.1)
        assert result["inference_calls"] == 1
    assert len(result["observations"]) == 3
    assert model.calls == 3


def test_expired_labels_are_not_emitted(scene):
    scene[20:40,50:70] = (180,180,0)
    model = Predictor()
    tracker = TemporalVision(model,budget=1,refresh_interval=.1,scan_interval=.1,label_ttl=.1)
    first = tracker.observe(scene,0)
    second = tracker.observe(scene,.1)
    assert len(first["observations"]) == len(second["observations"]) == 1
    assert second["observations"][0]["classification_fresh"]


def test_timestamp_validation(scene):
    tracker = TemporalVision(Predictor())
    tracker.observe(scene,1)
    for timestamp in (1,0,float("nan"),float("inf")):
        with pytest.raises(ValueError,match="timestamps"):
            tracker.observe(scene,timestamp)


def test_color_change_invalidates_identity(scene):
    model = Predictor()
    tracker = TemporalVision(model)
    tracker.observe(scene,0)
    changed = scene.copy()
    changed[20:40,20:40] = (180,0,180)
    result = tracker.observe(changed,.1)
    assert model.calls == 2
    assert all(row["classification_fresh"] for row in result["observations"])


def test_resolution_change_discards_old_tracks(scene):
    model = Predictor()
    tracker = TemporalVision(model)
    first = tracker.observe(scene,0)
    second = tracker.observe(cv2.resize(scene,(60,40)),.1)
    assert second["full_scan"] and model.calls == 2
    assert first["observations"][0]["track_id"] != second["observations"][0]["track_id"]


@pytest.mark.parametrize("kwargs",[dict(budget=1.5),dict(budget=0),dict(label_ttl=float("inf")),
                                    dict(refresh_interval=float("nan")),dict(scan_interval=2)])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        TemporalVision(Predictor(),**kwargs)


def test_confident_background_uses_bounded_slower_refresh(scene):
    model = Predictor()

    def background(rgb):
        model.calls += 1
        return dict(label="background",score=.99,accepted=False)

    model.predict = background
    tracker = TemporalVision(model)
    for timestamp in (0,.6,1.2):
        result = tracker.observe(scene,timestamp)
        assert result["target_lost"]
    assert model.calls == 1
    tracker.observe(scene,1.9)
    assert model.calls == 2


def test_background_appearance_change_reclassifies_immediately(scene):
    model = Predictor()

    def background(rgb):
        model.calls += 1
        return dict(label="background",score=.99,accepted=False)

    model.predict = background
    tracker = TemporalVision(model)
    tracker.observe(scene,0)
    changed = scene.copy()
    changed[20:40,20:40] = (180,0,180)
    tracker.observe(changed,.1)
    assert model.calls == 2
