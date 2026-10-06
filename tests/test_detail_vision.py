import numpy as np
import pytest

from dtr import detail_vision
from dtr.vision import observe


class RecordingPredictor:
    metadata = {"task": "goal"}

    def __init__(self):
        self.crops = []

    def predict(self, crop):
        self.crops.append(crop.copy())
        return dict(label="yellow_square", score=.9, accepted=True)


def test_mapped_geometry_and_real_detail(monkeypatch):
    candidate = dict(box=[2,3,6,7], crop_box=[1,2,7,8], area=16, color_group=1)
    monkeypatch.setattr(detail_vision, "proposals", lambda *a, **kw: [candidate])
    source = np.arange(16*16*3, dtype=np.uint8).reshape(16,16,3)
    scan = np.zeros((8,8,3), np.uint8)
    predictor = RecordingPredictor()
    got = detail_vision.observe_detail(source, predictor, scan_rgb=scan)
    row = got["observations"][0]
    assert row["box"] == [4,6,12,14]
    assert row["crop_box"] == [2,4,14,16]
    assert row["center_normalized"] == [0,.25]
    np.testing.assert_array_equal(predictor.crops[0], source[4:16,2:14])
    assert got["coordinate_space"] == "source_pixels"
    assert got["flight_commands"] is None


def test_same_resolution_preserves_original_path():
    rgb = np.zeros((80,80,3), np.uint8)
    rgb[10:45,20:60] = (255,255,0)
    a, b = RecordingPredictor(), RecordingPredictor()
    old = observe(rgb, a, profile="balloon_components")
    new = detail_vision.observe_detail(rgb, b, scan_rgb=rgb)
    assert old["accepted_count"] == new["accepted_count"] > 0
    for x, y in zip(old["observations"], new["observations"]):
        for key in x:
            assert x[key] == y[key]
    for x, y in zip(a.crops, b.crops):
        np.testing.assert_array_equal(x,y)


def test_reject_invalid_source():
    predictor = RecordingPredictor()
    with pytest.raises(ValueError, match="uint8"):
        detail_vision.observe_detail(np.zeros((8,8,3)), predictor)
    with pytest.raises(ValueError, match="smaller"):
        detail_vision.observe_detail(np.zeros((8,8,3), np.uint8), predictor,
                                    scan_rgb=np.zeros((16,16,3), np.uint8))
