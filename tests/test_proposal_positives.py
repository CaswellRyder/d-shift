import pytest

from dtr.data import sha256, write_json
from scripts.build_red_blue_proposal_positives import checked_positives, matched_target


def test_proposal_match_requires_unique_overlap_and_color():
    c = dict(box=[10, 10, 30, 30], crop_box=[8, 8, 32, 32], color_group=0)
    t = dict(scan_box=[10, 10, 30, 30], annotation_id=1, label="red_balloon")
    assert matched_target(c, [t]) == t
    assert matched_target(dict(c, color_group=1), [t]) is None
    assert matched_target(dict(c, box=[0, 0, 5, 5]), [t]) is None
    assert matched_target(c, [t, dict(t, annotation_id=2)]) is None
    assert matched_target(c, [t, dict(t, annotation_id=2, label="blue_balloon",
                                   scan_box=[8, 8, 20, 32])]) is None


def fixture(tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fixture")
    row = dict(path="crop.png", sha256=sha256(crop), label="blue_balloon", split="train",
               session="photo:a", source_image="balloon/train/a.jpg", source_sha256="a",
               annotation_id=1, box=[10, 10, 30, 30], crop_box=[8, 8, 32, 32],
               target_box=[10, 10, 30, 30])
    manifest = tmp_path / "manifest.json"
    write_json(manifest, dict(samples=[row]))
    queue = tmp_path / "review.json"
    write_json(queue, dict(base_manifest_sha256=sha256(manifest), samples=[row]))
    decision = tmp_path / "decision.json"
    write_json(decision, dict(queue_sha256=sha256(queue), admit=[0]))
    return row, manifest, queue, decision


def test_positive_review_and_pixels_bound_to_original_partition(tmp_path):
    row, manifest, queue, decision = fixture(tmp_path)
    assert checked_positives(tmp_path, decision, manifest) == [row]
    (tmp_path / "crop.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Invalid positive"):
        checked_positives(tmp_path, decision, manifest)


@pytest.mark.parametrize("changes", [dict(split="val"), dict(label="red_balloon"),
    dict(annotation_id=2), dict(source_image="balloon/val/a.jpg"), dict(session="other"),
    dict(path="../escape.png"), dict(box=[0, 0, 3, 3]), dict(target_box=[0, 0, 0, 0])])
def test_positive_admission_rejects_invalid_provenance(tmp_path, changes):
    row, manifest, queue, decision = fixture(tmp_path)
    write_json(queue, dict(base_manifest_sha256=sha256(manifest), samples=[dict(row, **changes)]))
    write_json(decision, dict(queue_sha256=sha256(queue), admit=[0]))
    with pytest.raises(ValueError, match="Invalid positive"):
        checked_positives(tmp_path, decision, manifest)


def test_unreviewed_or_modified_queue_rejected(tmp_path):
    row, manifest, queue, decision = fixture(tmp_path)
    write_json(queue, dict(base_manifest_sha256=sha256(manifest), samples=[row, row]))
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_positives(tmp_path, decision, manifest)
