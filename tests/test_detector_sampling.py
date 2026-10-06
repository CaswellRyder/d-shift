from copy import deepcopy

import pytest

from dtr.data import read_json, sha256, write_json
from dtr.detector_sampling import parse_rows, prepare_close_range


def source_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset.yaml").write_text("names: [a, b, c, d, e, f]\n")
    result = dict(classes=list("abcdef"), test_exported=False, test_evaluated=False,
                  deployment_approved=False, dataset_yaml_sha256=sha256(source / "dataset.yaml"),
                  splits={})
    for split in ("train", "valid"):
        records = []
        for name, labels in [("large.jpg", "4 .5 .5 .6 .4\n0 .1 .1 .1 .1\n"),
                             ("small.jpg", "0 .5 .5 .1 .1\n"), ("negative.jpg", "")]:
            image = source / "images" / split / name
            label = source / "labels" / split / name.replace(".jpg", ".txt")
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"{split}/{name}".encode())
            label.write_text(labels)
            records.append(dict(file=name, sha256=sha256(image), labels_sha256=sha256(label)))
        result["splits"][split] = dict(images=records, frames=3, counts=dict(a=2, e=1))
    write_json(source / "receipt.json", result)
    return source, result


def test_repeats_train_only_preserves_all_labels_backgrounds_and_valid(tmp_path):
    source, original = source_fixture(tmp_path)
    output = tmp_path / "derived"
    result = prepare_close_range(source, output)
    assert result["splits"]["valid"] == original["splits"]["valid"]
    assert result["splits"]["train"]["frames"] == 6
    assert result["splits"]["train"]["counts"] == dict(a=5, e=4)
    assert result["training_exposure"]["selected_files"] == ["large.jpg"]
    assert result["training_exposure"]["unique_training_frames"] == 3
    assert (output / "labels/train/close-repeat-3--large.txt").read_text() == (
        source / "labels/train/large.txt").read_text()
    assert (output / "images/train/negative.jpg").exists()
    assert not (output / "images/test").exists()
    assert read_json(source / "receipt.json") == original
    assert not result["deployment_approved"]
    with pytest.raises(FileExistsError):
        prepare_close_range(source, output)
    with pytest.raises(ValueError, match="original"):
        prepare_close_range(output, tmp_path / "again")


def test_refuses_source_tampering_before_creating_output(tmp_path):
    source, _ = source_fixture(tmp_path)
    (source / "labels/train/large.txt").write_text("4 .5 .5 .8 .8\n")
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="changed"):
        prepare_close_range(source, output)
    assert not output.exists()


def test_refuses_cross_split_duplicate(tmp_path):
    source, receipt = source_fixture(tmp_path)
    target = source / "images/valid/large.jpg"
    target.write_bytes((source / "images/train/large.jpg").read_bytes())
    receipt["splits"]["valid"]["images"][0]["sha256"] = sha256(target)
    write_json(source / "receipt.json", receipt)
    with pytest.raises(ValueError, match="Cross-split"):
        prepare_close_range(source, tmp_path / "derived")


def test_rejects_test_scope_and_unsafe_names(tmp_path):
    source, original = source_fixture(tmp_path)
    for key in ("test_exported", "test_evaluated", "deployment_approved"):
        receipt = deepcopy(original)
        receipt[key] = True
        write_json(source / "receipt.json", receipt)
        with pytest.raises(ValueError, match="unapproved"):
            prepare_close_range(source, tmp_path / "derived")
    receipt = deepcopy(original)
    receipt["splits"]["train"]["images"][0]["file"] = "../escape.jpg"
    write_json(source / "receipt.json", receipt)
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_close_range(source, tmp_path / "derived")


def test_threshold_and_label_validation(tmp_path):
    source, _ = source_fixture(tmp_path)
    for threshold in (0, 1, float("nan")):
        with pytest.raises(ValueError, match="threshold"):
            prepare_close_range(source, tmp_path / "derived", threshold=threshold)
    for invalid in ("0 .5 .5 nan .2", "6 .5 .5 .1 .1", "1.5 .5 .5 .1 .1",
                    "0 .5 .5 0 .2", "0 .5 .5 .1"):
        with pytest.raises(ValueError):
            parse_rows(invalid, 6)
    with pytest.raises(ValueError, match="No qualifying"):
        prepare_close_range(source, tmp_path / "derived", threshold=.9)
    with pytest.raises(ValueError, match="separate"):
        prepare_close_range(source, source / "nested")
