import io
import json
import zipfile

from PIL import Image
import pytest

from dtr.data import read_json, sha256, write_json
from scripts.audit_public_balloon_export import audit, bounded_read, checked_member, source_group


def test_source_group_is_only_a_filename_heuristic():
    assert source_group("train/frame_jpg.rf.ab1234.jpg") == source_group("test/frame_jpg.rf.cd1234.jpg")
    assert source_group("frame2_jpg.rf.ab1234.jpg") != source_group("frame_jpg.rf.ab1234.jpg")


@pytest.mark.parametrize("name", ["/absolute.jpg", "train/../test/a.jpg", "train\\a.jpg"])
def test_archive_paths_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        checked_member(name)


def test_audit_reads_only_training_pixels_and_preserves_quarantine(tmp_path):
    raw = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(raw, format="PNG")
    root = tmp_path / "raw"
    root.mkdir()
    path = root / "dataset.zip"
    with zipfile.ZipFile(path, "w") as z:
        for split, suffix in (("train", "ab12"), ("valid", "cd34"), ("test", "ef56")):
            filename = f"same.rf.{suffix}.png"
            z.writestr(f"{split}/{filename}", raw.getvalue() if split == "train" else b"NOT AN IMAGE")
            z.writestr(f"{split}/_annotations.coco.json", json.dumps(dict(
                images=[dict(id=1, file_name=filename, width=32, height=32)],
                categories=[dict(id=1, name="blue_ballon")],
                annotations=[dict(id=1, image_id=1, category_id=1, bbox=[2,2,24,24])])))
    write_json(root / "download-receipt.json", dict(source="fixture", archive_sha256=sha256(path)))
    audit(root, tmp_path / "out")
    report = read_json(tmp_path / "out/audit.json")
    assert report["cross_split_basename_groups"] == {"same.png": ["test", "train", "valid"]}
    assert len(report["previews"]) == 1
    assert not report["training_approved"] and not report["label_mapping_approved"]
    with zipfile.ZipFile(path) as z:
        with pytest.raises(ValueError, match="read bound"):
            bounded_read(z, "train/same.rf.ab12.png", 1)
