import cv2
import numpy as np
import pytest

from dtr.data import read_json, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.build_native_balloon_regions import build
from scripts.research_balloon_search import candidate, experimental
from scripts.research_balloon_search_fast import search_fast


@pytest.fixture(scope="module", params=[False, True], ids=["cast", "direct"])
def native(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("region-library") / "regions.so"
    build("src/dtr/native_balloon_regions.c", path)
    return NativeBalloonRegions(path, direct=request.param)


def reference(plane, points):
    if plane[points[:, 1], points[:, 0]].mean() < 15:
        return None
    return candidate(points, 0, (240, 320, 3))


def assert_candidate_parity(native, plane, points):
    before = points.copy()
    reduced = native.reduce(plane, points)
    result = candidate(reduced, 0, (240, 320, 3)) if len(reduced) else None
    assert result == reference(plane, points)
    assert np.array_equal(points, before)
    if len(reduced):
        assert cv2.contourArea(cv2.convexHull(reduced)) == cv2.contourArea(cv2.convexHull(points))
        assert cv2.boundingRect(reduced) == cv2.boundingRect(points)


def test_random_clouds_duplicates_rows_and_geometry_are_exact(native):
    rng = np.random.default_rng(1337)
    for _ in range(300):
        lo = rng.integers([0, 0], [310, 230])
        hi = rng.integers(lo + 1, [321, 241])
        points = rng.integers(lo, hi, (int(rng.integers(1, 25001)), 2), dtype=np.int32)
        plane = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        assert_candidate_parity(native, plane, points)


@pytest.mark.parametrize("value", [0, 14, 15, 16, 255])
def test_mean_threshold_and_dense_region(native, value):
    yy, xx = np.mgrid[1:151, 4:164]
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.int32)
    assert_candidate_parity(native, np.full((240, 320), value, np.uint8), points)


@pytest.mark.parametrize("w,h", [(3, 4), (4, 4), (12, 4), (13, 4), (240, 240), (241, 240)])
def test_geometry_rejection_boundaries(native, w, h):
    points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.int32)
    assert_candidate_parity(native, np.full((240, 320), 30, np.uint8), points)


@pytest.mark.parametrize("xy", [(-1, 2), (320, 2), (2, -1), (2, 240), (2**31 - 1, 1)])
def test_c_checks_coordinates_before_indexing(native, xy):
    with pytest.raises(ValueError, match="coordinates"):
        native.reduce(np.zeros((240, 320), np.uint8), np.array([xy], np.int32))


def test_wrapper_rejects_wrong_buffer_contracts(native):
    plane = np.zeros((240, 320), np.uint8)
    points = np.zeros((5, 2), np.int32)
    for bad in (plane.astype(np.int32), plane.T, plane[:, ::2]):
        with pytest.raises(ValueError, match="plane"):
            native.reduce(bad, points)
    for bad in (points.astype(np.int64), points.T, points[::2], points[:0], points.reshape(-1)):
        with pytest.raises(ValueError, match="points"):
            native.reduce(plane, bad)


def test_receipt_hash_mismatch_fails(native, tmp_path):
    path = tmp_path / "bad.so"
    path.write_bytes(native.path.read_bytes())
    receipt = read_json(native.path.with_suffix(".json"))
    receipt["sha256"] = "wrong"
    write_json(path.with_suffix(".json"), receipt)
    with pytest.raises(ValueError, match="contract/hash"):
        NativeBalloonRegions(path)


@pytest.mark.parametrize("mode", [None, 1, "true", np.bool_(True)])
def test_pointer_mode_requires_explicit_boolean(native, mode):
    with pytest.raises(ValueError, match="explicit boolean"):
        NativeBalloonRegions(native.path, direct=mode)


def test_full_search_parity_on_generated_scenes(native):
    rng = np.random.default_rng(82)
    for i in range(12):
        rgb = rng.integers(0, 30, (240, 320, 3), dtype=np.uint8)
        for _ in range(i):
            center = tuple(map(int, rng.integers([0, 0], [320, 240])))
            color = tuple(map(int, rng.integers(0, 256, 3)))
            cv2.circle(rgb, center, int(rng.integers(2, 65)), color, -1)
        assert search_fast(rgb, native) == experimental(rgb, "mser")


def test_bright_only_keeps_isolated_disks_but_is_not_semantically_exact(native):
    rgb = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(rgb, (80, 100), 18, (200, 0, 0), -1)
    cv2.circle(rgb, (220, 100), 18, (0, 0, 200), -1)
    assert search_fast(rgb, native, bright_only=True) == search_fast(rgb, native)
    # A dark red disk on a brighter red field is a real proposal counterexample.
    # Generated geometry is NOT a labeled real balloon or qualification sample.
    rgb[:] = (230, 0, 0)
    cv2.circle(rgb, (160, 120), 18, (100, 0, 0), -1)
    assert len(search_fast(rgb, native)) == 1
    assert search_fast(rgb, native, bright_only=True) == []
    with pytest.raises(ValueError, match="explicit boolean"):
        search_fast(rgb, native, bright_only=1)


def test_native_output_survives_next_call(native):
    plane = np.full((240, 320), 30, np.uint8)
    points = np.array([[0, 0], [10, 10], [4, 2], [5, 2]], np.int32)
    first = native.reduce(plane, points)
    saved = first.copy()
    native.reduce(plane, points + 5)
    assert np.array_equal(first, saved)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_mixed_values_at_mean_threshold(native, delta):
    points = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.int32)
    plane = np.zeros((240, 320), np.uint8)
    plane[10, 10] = 60 + delta
    assert_candidate_parity(native, plane, points)


def test_maximum_output_and_buffer_contract(native):
    yy, xx = np.mgrid[:240, :240]
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.int32)
    plane = np.full((240, 320), 255, np.uint8)
    reduced = native.reduce(plane, points)
    assert len(reduced) == 480
    assert_candidate_parity(native, plane, points)
    with pytest.raises(ValueError, match="points"):
        native.reduce(plane, np.zeros((76801, 2), np.int32))
