import importlib.util
from pathlib import Path

from PIL import Image
import pytest
import yaml

from dtr.data import read_json, write_json, sha256

spec = importlib.util.spec_from_file_location(
    "prepare_detector", Path(__file__).parents[1] / "scripts/prepare_goal_detector.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_yolo_box_preserves_small_targets_and_clips_bounds():
    assert module.yolo_box([1, 1, 2, 2], 10, 10) == [0.2, 0.2, 0.2, 0.2]
    assert module.yolo_box([-1, -1, 3, 3], 10, 10) == [0.1, 0.1, 0.2, 0.2]
    for box in ([0, 0, 0, 1], [12, 0, 1, 1], [float("nan"), 0, 1, 1]):
        with pytest.raises(ValueError):
            module.yolo_box(box, 10, 10)


def test_detector_export_leaves_test_unread_and_uses_correct_class_order(tmp_path):
    cfg = read_json("configs/goal.json")
    labels = cfg["classes"][1:]
    # Use the canonical aliases already present in project config.
    aliases = {next(k for k, v in cfg["aliases"].items() if v == c): c for c in labels}
    categories = [dict(id=i, name=k) for i, k in enumerate(aliases)]
    source = tmp_path / "coco"
    for split, shade in (("train", 30), ("valid", 80)):
        folder = source / split
        folder.mkdir(parents=True)
        Image.new("RGB", (10, 10), (shade, 0, 0)).save(folder / "frame.png")
        doc = dict(images=[dict(id=0, file_name="frame.png", width=10, height=10)],
                   categories=categories,
                   annotations=[dict(image_id=0, category_id=i, bbox=[1, 1, 2, 2])
                                for i in range(6)])
        write_json(folder / "_annotations.coco.json", doc)
    (source / "test").mkdir()
    (source / "test/_annotations.coco.json").write_text("MUST NOT BE READ")
    output = tmp_path / "export"
    receipt = module.prepare(source, output, cfg, [])
    assert set(receipt["splits"]) == {"train", "valid"}
    assert not receipt["test_exported"]
    exported = yaml.safe_load((output / "dataset.yaml").read_text())
    assert "test" not in exported and list(exported["names"].values()) == labels
    assert len((output / "labels/train/frame.txt").read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        module.prepare(source, output, cfg, [])


@pytest.fixture
def duplicate_review(tmp_path):
    cfg = read_json("configs/goal.json")
    labels = cfg["classes"][1:]
    categories = [dict(id=i, name=next(k for k, v in cfg["aliases"].items() if v == name))
                  for i, name in enumerate(labels)]
    source = tmp_path / "source"
    for split, shade in (("train", 30), ("valid", 80)):
        folder = source / split
        folder.mkdir(parents=True)
        Image.new("RGB", (100, 100), (shade, 0, 0)).save(folder / "frame.png")
        anns = [dict(id=i, image_id=0, category_id=i, bbox=[i*10, 0, 5, 5])
                for i in range(6)]
        anns.append(dict(id=99, image_id=0, category_id=1, bbox=[0, 0, 5, 5]))
        write_json(folder / "_annotations.coco.json", dict(
            images=[dict(id=0, file_name="frame.png", width=100, height=100)],
            annotations=anns, categories=categories))
    review = dict(scope="train_only", reviewer="test fixture",
                  annotation_sha256=sha256(source / "train/_annotations.coco.json"),
                  decisions=[dict(file="frame.png", image_sha256=sha256(source / "train/frame.png"),
                                  drop_annotation_id=99, retain_annotation_id=0,
                                  retain_label=labels[0])])
    path = tmp_path / "review.json"
    write_json(path, review)
    return source, cfg, path, review


def test_review_applies_only_to_fresh_training_export(duplicate_review, tmp_path):
    source, cfg, path, review = duplicate_review
    original_sha = sha256(source / "train/_annotations.coco.json")
    result = module.prepare(source, tmp_path / "reviewed", cfg, [], path)
    assert len((tmp_path / "reviewed/labels/train/frame.txt").read_text().splitlines()) == 6
    assert len((tmp_path / "reviewed/labels/valid/frame.txt").read_text().splitlines()) == 7
    assert result["training_label_review"]["dropped_annotation_ids"] == [99]
    assert sha256(source / "train/_annotations.coco.json") == original_sha


@pytest.mark.parametrize("mutation", ["scope", "annotations", "image", "same_id", "wrong_label"])
def test_review_fails_closed_on_scope_or_binding_change(duplicate_review, mutation):
    source, cfg, path, review = duplicate_review
    if mutation == "scope":
        review["scope"] = "valid"
    elif mutation == "annotations":
        review["annotation_sha256"] = "changed"
    elif mutation == "image":
        review["decisions"][0]["image_sha256"] = "changed"
    elif mutation == "same_id":
        review["decisions"][0]["retain_annotation_id"] = 99
    else:
        review["decisions"][0]["retain_label"] = "wrong_class"
    write_json(path, review)
    with pytest.raises(ValueError):
        module.reviewed_training_drops(source, cfg, path)


@pytest.fixture
def color_review(duplicate_review, tmp_path):
    source, cfg, _, _ = duplicate_review
    decision = dict(file="frame.png", image_sha256=sha256(source / "train/frame.png"),
                    annotation_id=99, original_label="orange_square", original_bbox=[0, 0, 5, 5],
                    action="relabel", replacement_label="yellow_square", reason="Visual review")
    review = dict(scope="train_only", reviewer="test fixture", decisions=[decision],
                  annotation_sha256=sha256(source / "train/_annotations.coco.json"))
    path = tmp_path / "color-review.json"
    write_json(path, review)
    return source, cfg, path, review


@pytest.mark.parametrize("action", ["relabel", "remove_non_goal"])
def test_visual_correction_is_train_only_and_keeps_source(color_review, tmp_path, action):
    source, cfg, path, review = color_review
    original = sha256(source / "train/_annotations.coco.json")
    if action == "remove_non_goal":
        review["decisions"][0].update(action=action, replacement_label=None)
    write_json(path, review)
    (source / "test").mkdir()
    (source / "test/_annotations.coco.json").write_text("DO NOT READ")
    result = module.prepare(source, tmp_path / "corrected", cfg, [], corrections_path=path)
    train = (tmp_path / "corrected/labels/train/frame.txt").read_text().splitlines()
    valid = (tmp_path / "corrected/labels/valid/frame.txt").read_text().splitlines()
    assert len(valid) == 7 and valid[-1].startswith("1 ")
    assert len(train) == (7 if action == "relabel" else 6)
    if action == "relabel":
        assert train[-1].startswith("4 ") and train[-1][2:] == valid[-1][2:]
    assert result["training_label_corrections"]["scope"] == "train_only"
    assert sha256(source / "train/_annotations.coco.json") == original


@pytest.mark.parametrize("mutation", ["scope", "checksum", "image", "bbox", "label", "unknown",
                                     "duplicate", "path", "reserved", "action", "replacement"])
def test_visual_correction_rejects_invalid_binding(color_review, mutation):
    source, cfg, path, review = color_review
    d = review["decisions"][0]
    if mutation == "scope":
        review["scope"] = "valid"
    elif mutation == "checksum":
        review["annotation_sha256"] = "changed"
    elif mutation == "image":
        d["image_sha256"] = "changed"
    elif mutation == "bbox":
        d["original_bbox"] = [1, 1, 5, 5]
    elif mutation == "label":
        d["original_label"] = "yellow_square"
    elif mutation == "unknown":
        d["annotation_id"] = -1
    elif mutation == "duplicate":
        review["decisions"].append(d.copy())
    elif mutation == "path":
        d["file"] = "../valid/frame.png"
    elif mutation == "reserved":
        d["file"] = "Highbay_frame.png"
    elif mutation == "action":
        d["action"] = "guess"
    else:
        d["replacement_label"] = "background"
    write_json(path, review)
    with pytest.raises(ValueError):
        module.reviewed_training_corrections(source, cfg, path)


def test_visual_correction_cannot_override_duplicate_review(color_review, duplicate_review, tmp_path):
    source, cfg, path, _ = color_review
    _, _, duplicates, _ = duplicate_review
    with pytest.raises(ValueError, match="conflict"):
        module.prepare(source, tmp_path / "conflicting", cfg, [], duplicates, path)
    assert not (tmp_path / "conflicting").exists()
