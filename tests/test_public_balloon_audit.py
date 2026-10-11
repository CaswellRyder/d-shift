import io
import json
import zipfile

from PIL import Image
import pytest

from dtr.data import read_json, sha256, write_json
from scripts.audit_public_balloon_export import (
    audit, bounded_read, checked_member, excluded_image_keys, original_image_key, source_group,
)


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


@pytest.mark.parametrize("name", ["train/Photo_JPG.rf.ab12.jpg", "Photo.jpg", "Photo_png.png"])
def test_original_key_recognizes_common_reexports(name):
    assert original_image_key(name) == "photo"


def test_known_source_alias_is_excluded_before_any_pixels_are_read(tmp_path):
    reference = tmp_path/"old.zip"
    with zipfile.ZipFile(reference,"w") as z:
        z.writestr("balloon/val/photo.jpg", b"REFERENCE PIXELS MUST NOT BE DECODED")
        z.writestr("balloon/val/._photo.jpg", b"MAC METADATA")
    raw = tmp_path/"raw"
    raw.mkdir()
    archive = raw/"dataset.zip"
    with zipfile.ZipFile(archive,"w") as z:
        for split in ("train","valid","test"):
            # Deliberately omit image members. Metadata exclusion must precede
            # all reads, including reads of the re-exported training image.
            z.writestr(f"{split}/_annotations.coco.json",json.dumps(dict(
                images=[dict(id=1,file_name="photo_jpg.rf.ab12.jpg",width=32,height=32)],
                categories=[dict(id=1,name="blue-balloon")],
                annotations=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,32,32])])) )
    write_json(raw/"download-receipt.json",dict(source="fixture",archive_sha256=sha256(archive)))
    out = tmp_path/"out"
    audit(raw,out,[(reference,sha256(reference))])
    doc = read_json(out/"audit.json")
    assert len(doc["known_source_aliases"]) == 3
    assert doc["previews"] == [] and not list(out.glob("*.jpg"))
    assert doc["exclusion_archives"][0]["image_members"] == 1
    assert not doc["training_approved"]


def test_exclusion_archive_requires_expected_hash(tmp_path):
    archive = tmp_path/"reference.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("a.jpg", b"unused")
    with pytest.raises(ValueError,match="hash mismatch"):
        excluded_image_keys([(archive,"0"*64)])
