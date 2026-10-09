from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dtr.data import load_rgb, read_json, sha256, synthetic, write_json
from scripts import build_paired_balloon_augmentation as augmentation
from scripts.build_paired_balloon_augmentation import build, transform
from scripts.distill_pi_student import checked_cache


def test_symmetries_keep_every_pixel_and_complete_rectangular_context():
    rgb = np.arange(5*7*3, dtype=np.uint8).reshape(5, 7, 3)
    views = []
    for code in range(8):
        result = transform(rgb, code)
        assert sorted(map(tuple, result.reshape(-1, 3))) == sorted(map(tuple, rgb.reshape(-1, 3)))
        assert result.shape == ((7, 5, 3) if code % 2 else (5, 7, 3))
        assert result.flags.c_contiguous
        views.append(result.tobytes())
    assert len(set(views)) == 8
    np.testing.assert_array_equal(transform(rgb, 0), rgb)


@pytest.mark.parametrize("code", [True, -1, 8, 1.5])
def test_invalid_symmetries_rejected(code):
    with pytest.raises(ValueError):
        transform(np.zeros((5, 7, 3), np.uint8), code)


@pytest.fixture
def manifest(tmp_path):
    config = read_json("configs/balloon-red-blue.json")
    path = Path(synthetic(tmp_path / "base", config, per_class=1))
    doc = read_json(path)
    doc.update(synthetic=False, training_approved=True)
    for row in doc["samples"]:
        row["source_image"] = f"{row['split']}/{row['label']}.png"
        if row["split"] == "test":
            (path.parent / row["path"]).unlink()
    write_json(path, doc)
    return path


def test_paired_views_match_counts_keep_holdout_and_bind_cache(manifest, tmp_path):
    augmentation, control = tmp_path / "aug", tmp_path / "control"
    a = build(manifest, augmentation)
    b = build(manifest, control, control=True)
    assert a["counts"] == b["counts"]
    assert a["added_entries"] == b["added_entries"] == 3
    assert a["manifest_sha256"] != b["manifest_sha256"] != sha256(manifest)
    base, adoc, bdoc = [read_json(p) for p in
        (manifest, augmentation / "manifest.json", control / "manifest.json")]
    before = [r for r in base["samples"] if r["split"] != "train"]
    for root, doc in ((augmentation, adoc), (control, bdoc)):
        assert [r for r in doc["samples"] if r["split"] != "train"] == [
            dict(r, path="base/" + r["path"]) for r in before]
        assert all(not (root / r["path"]).exists() for r in doc["samples"] if r["split"] == "test")
    for ar, br in zip(adoc["samples"][len(base["samples"]):], bdoc["samples"][len(base["samples"]):]):
        assert ar["proposed_augmentation_code"] == br["proposed_augmentation_code"]
        assert ar["augmentation_code"] != 0 and br["augmentation_code"] == 0
        original = base["samples"][ar["augmentation_parent_row"]]
        with Image.open(manifest.parent / original["path"]) as im:
            expected = Image.fromarray(transform(np.asarray(im.convert("RGB")), ar["augmentation_code"]))
        for size in (64, 96):
            np.testing.assert_array_equal(load_rgb(augmentation / ar["path"], size),
                np.asarray(expected.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32))
    cache = tmp_path / "cache"
    write_json(cache / "provenance.json", dict(teacher_targets_kind="teacher_logits",
        teacher_sha256="teacher", manifest_sha256=sha256(manifest), config=dict(classes=base["classes"])))
    # Reject before attempting to read a nonexistent parent target file.
    with pytest.raises(ValueError, match="identity"):
        checked_cache(cache, a["manifest_sha256"], "teacher", [], base["classes"])
    with pytest.raises(FileExistsError):
        build(manifest, augmentation)


@pytest.mark.parametrize("bad", ["img", "missing_source", "holdout_source", "approval", "synthetic", "repeated"])
def test_unsafe_sources_and_approval_rejected_before_output(manifest, tmp_path, bad):
    doc = read_json(manifest)
    row = next(r for r in doc["samples"] if r["split"] == "train")
    if bad == "img":
        row["source_image"] = "train/IMG_1234_jpg.rf.abc.jpg"
    elif bad == "missing_source":
        row.pop("source_image")
    elif bad == "holdout_source":
        row["source_image"] = next(r["source_image"] for r in doc["samples"] if r["split"] == "test")
    elif bad == "approval":
        doc["training_approved"] = False
    elif bad == "synthetic":
        doc["synthetic"] = True
    else:
        doc["paired_augmentation"] = {}
    write_json(manifest, doc)
    with pytest.raises(ValueError):
        build(manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("seed", [True, -1, 2**32, 4.2])
def test_invalid_seed_rejected(seed, tmp_path):
    with pytest.raises(ValueError, match="seed"):
        build("missing", tmp_path / "out", seed=seed)


def test_build_is_deterministic(manifest, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert build(manifest, a) == build(manifest, b)
    assert read_json(a / "manifest.json") == read_json(b / "manifest.json")


def test_conflicting_parent_labels_fail_before_output(manifest, tmp_path):
    doc = read_json(manifest)
    first, second = [r for r in doc["samples"] if r["split"] == "train"][:2]
    second.update(path=first["path"], sha256=first["sha256"])
    write_json(manifest, doc)
    with pytest.raises(ValueError, match="Conflicting labels"):
        build(manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_failed_output_validation_stays_unapproved(manifest, tmp_path, monkeypatch):
    original_validate = augmentation.validate

    def fail_output(path, *args, **kwargs):
        if Path(path) != manifest:
            raise ValueError("output validation failed")
        return original_validate(path, *args, **kwargs)

    monkeypatch.setattr(augmentation, "validate", fail_output)
    with pytest.raises(ValueError, match="output validation"):
        build(manifest, tmp_path / "out")
    assert read_json(tmp_path / "out/manifest.json")["training_approved"] is False
