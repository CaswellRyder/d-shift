import numpy as np
import pytest

from scripts.research_blue_bright_search import impossible_geometry, search


def test_prefilter_matches_existing_size_limits():
    assert impossible_geometry(3, 4)
    assert impossible_geometry(13, 4)
    assert impossible_geometry(4, 13)
    assert impossible_geometry(320, 240)
    assert not impossible_geometry(12, 4)
    assert not impossible_geometry(240, 240)


def test_explicit_policy_and_rgb_contract():
    with pytest.raises(ValueError):
        search(np.zeros((240, 320, 3), np.uint8), None, 'true')
    with pytest.raises(ValueError):
        search(np.zeros((120, 160, 3), np.uint8), None, True)


def test_only_blue_uses_second_pass(monkeypatch):
    import scripts.research_blue_bright_search as module
    detectors = []
    class Detector:
        def __init__(self):
            self.second = False
        def setPass2Only(self, value):
            self.second = value
        def detectRegions(self, plane):
            return [], []
    def create(*args):
        d = Detector()
        detectors.append(d)
        return d
    monkeypatch.setattr(module.cv2, 'MSER_create', create)
    assert search(np.zeros((240, 320, 3), np.uint8), None, True) == []
    assert [d.second for d in detectors] == [False, True]
