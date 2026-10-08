import cv2
import numpy as np
import pytest
from pathlib import Path
import shutil

from dtr.color_lookup import GoalColorLookup, RedBlueColorLookup
from dtr.data import sha256, write_json
from dtr.vision import RED_BLUE_PROFILE, color_masks, proposals
from scripts.build_red_blue_lookup import rgb_slice
from scripts.build_native_color_lookup import build


@pytest.fixture(scope="module")
def red_blue_lookup(tmp_path_factory):
    root = tmp_path_factory.mktemp("red-blue-lookup")
    path = root / "colors.bin"
    values = np.empty((256, 256, 256), np.uint8)
    for red in range(256):
        a, b = color_masks(rgb_slice(red), "balloon", RED_BLUE_PROFILE)
        values[red] = (a//255) | ((b//255) << 1)
    path.write_bytes(values.tobytes())
    write_json(path.with_suffix(".json"), dict(contract=RedBlueColorLookup.contract, sha256=sha256(path)))
    return path


def test_red_blue_lookup_exact_masks_and_proposal_order(red_blue_lookup, monkeypatch):
    rng = np.random.default_rng(321)
    fixtures = [rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
                rng.integers(0, 256, (32, 62, 3), dtype=np.uint8)[:, ::2]]
    scene = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(scene, (50, 60), 20, (220, 10, 10), -1)
    cv2.circle(scene, (150, 60), 20, (10, 10, 220), -1)
    cv2.circle(scene, (50, 60), 6, (240, 240, 240), -1)
    fixtures.append(scene)
    for rgb in fixtures:
        before = rgb.copy()
        monkeypatch.delenv("DTR_RED_BLUE_COLOR_LOOKUP", raising=False)
        expected_masks = color_masks(rgb, "balloon", RED_BLUE_PROFILE)
        expected_candidates = proposals(rgb, "balloon", profile=RED_BLUE_PROFILE)
        legacy = color_masks(rgb, "balloon", "balloon_components")
        monkeypatch.setenv("DTR_RED_BLUE_COLOR_LOOKUP", str(red_blue_lookup))
        for a, b in zip(expected_masks, color_masks(rgb, "balloon", RED_BLUE_PROFILE)):
            np.testing.assert_array_equal(a, b)
        assert proposals(rgb, "balloon", profile=RED_BLUE_PROFILE) == expected_candidates
        for a, b in zip(legacy, color_masks(rgb, "balloon", "balloon_components")):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(rgb, before)


def test_lookup_rejects_wrong_taxonomy_contract(red_blue_lookup):
    with pytest.raises(ValueError, match="contract"):
        GoalColorLookup(red_blue_lookup)


def test_requested_lookup_never_silently_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("DTR_RED_BLUE_COLOR_LOOKUP", str(tmp_path / "absent.bin"))
    with pytest.raises(FileNotFoundError):
        color_masks(np.zeros((8, 8, 3), np.uint8), "balloon", RED_BLUE_PROFILE)


def test_native_lookup_exhaustive_parity_and_strided_inputs(red_blue_lookup, tmp_path, monkeypatch):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    source = Path(__file__).parents[1] / "src/dtr/native_color_lookup.c"
    library = tmp_path / "lookup.so"
    build(source, library, compiler)
    from dtr.native_color_lookup import NativeColorLookup
    native = NativeColorLookup(library)
    lookup = RedBlueColorLookup(red_blue_lookup)
    for red in range(256):
        rgb = rgb_slice(red)
        expected = color_masks(rgb, "balloon", RED_BLUE_PROFILE)
        for got, wanted in zip(native.masks(rgb, lookup.values), expected):
            np.testing.assert_array_equal(got, wanted)
    rgb = rgb_slice(170)[::2, ::3, ::-1]
    for got, wanted in zip(native.masks(rgb, lookup.values), lookup.masks(rgb)):
        np.testing.assert_array_equal(got, wanted)
    with pytest.raises(ValueError, match="uint8"):
        native.masks(rgb.astype(np.float32), lookup.values)
    with pytest.raises(ValueError, match="table"):
        native.masks(rgb, np.zeros(32, np.uint8))
    monkeypatch.setenv("DTR_RED_BLUE_LOOKUP_LIBRARY", str(library))
    for got, wanted in zip(lookup.masks(rgb), color_masks(rgb, "balloon", RED_BLUE_PROFILE)):
        np.testing.assert_array_equal(got, wanted)
