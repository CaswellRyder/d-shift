import numpy as np
import pytest

from dtr.data import sha256, write_json
from scripts.build_indoor_balloon_training import admitted
from scripts.prepare_indoor_balloon_review import family, fingerprint, holdout_distance


def test_frame_families_do_not_silently_admit_stock_or_other_names():
    assert family("frame_0012_jpg.rf.abcd.jpg") == "indoor-frame-4-digits"
    assert family("frame_000012_jpg.rf.abcd.jpg") == "indoor-frame-6-digits"
    for name in ("gettyimages-123.jpg", "frame_12_jpg.rf.abcd.jpg", "IMG_5123_JPG.rf.abcd.jpg"):
        assert family(name) is None


def test_duplicate_screen_catches_all_flip_orientations():
    rgb = np.random.default_rng(1).integers(0,256,(64,64,3),dtype=np.uint8)
    hashes = [fingerprint(rgb)]
    for view in (rgb, rgb[::-1], rgb[:,::-1], rgb[::-1,::-1]):
        assert holdout_distance(view,hashes) == 0


def queue_fixture(tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fixture")
    base = tmp_path / "base.json"
    write_json(base,dict(classes=["background","red_balloon","blue_balloon"],
                         samples=[dict(split="test",sha256="reserved")]))
    row = dict(id=0,path="crop.png",sha256=sha256(crop),frame_id=0,split="train",
               label="blue_balloon",session="engdes2:indoor-frame-4-digits")
    frame = dict(holdout_hamming=20,crop_ids=[0])
    queue = tmp_path / "review.json"
    write_json(queue,dict(base_manifest_sha256=sha256(base),samples=[row],frames=[frame]))
    review = tmp_path / "decision.json"
    write_json(review,dict(queue_sha256=sha256(queue),admit=[0],exclude=[]))
    return base,row,frame,queue,review


def test_admission_is_explicit_and_hash_bound(tmp_path):
    base,row,frame,queue,review = queue_fixture(tmp_path)
    assert admitted(tmp_path,review,base) == [row]
    write_json(review,dict(queue_sha256=sha256(queue),admit=[0],exclude=[0]))
    with pytest.raises(ValueError,match="Invalid admission"):
        admitted(tmp_path,review,base)
    write_json(review,dict(queue_sha256="changed",admit=[0],exclude=[]))
    with pytest.raises(ValueError,match="frozen inputs"):
        admitted(tmp_path,review,base)


@pytest.mark.parametrize("changes", [dict(split="test"),dict(label="unknown"),
    dict(session="independent-holdout"),dict(sha256="reserved"),dict(path="../escape.png"),
    dict(source_image="train/IMG_5123_JPG.rf.abcd.jpg")])
def test_admission_rejects_unscoped_crops(tmp_path,changes):
    base,row,frame,queue,review = queue_fixture(tmp_path)
    write_json(queue,dict(base_manifest_sha256=sha256(base),samples=[dict(row,**changes)],frames=[frame]))
    write_json(review,dict(queue_sha256=sha256(queue),admit=[0],exclude=[]))
    with pytest.raises(ValueError,match="Invalid external"):
        admitted(tmp_path,review,base)


@pytest.mark.parametrize("source,session,digest,allowed", [
    ("train/a", "train-session", "new", True),
    ("train/b", "train-session", "new", False),
    ("train/a", "test-session", "new", False),
    ("train/a", "train-session", "positive", False),
    ("train/a", "train-session", "heldout", False),
])
def test_hard_negatives_require_original_training_source_and_no_label_conflict(
        tmp_path, monkeypatch, source, session, digest, allowed):
    from scripts import refine_pi_student
    from scripts.build_indoor_balloon_training import checked_hard_negatives
    base = dict(samples=[
        dict(source_image="train/a", session="train-session", split="train",
             label="red_balloon", sha256="positive"),
        dict(source_image="val/a", session="val-session", split="val",
             label="background", sha256="heldout")])
    manifest = tmp_path / "manifest.json"
    write_json(manifest, base)
    rows = [dict(source_image=source, session=session, sha256=digest)]
    monkeypatch.setattr(refine_pi_student, "reviewed_negatives", lambda *args: rows)
    if allowed:
        assert checked_hard_negatives(base, manifest, tmp_path, "unused") == rows
    else:
        with pytest.raises(ValueError, match="outside original"):
            checked_hard_negatives(base, manifest, tmp_path, "unused")


def test_near_holdout_is_not_training_material(tmp_path):
    base,row,frame,queue,review = queue_fixture(tmp_path)
    write_json(queue,dict(base_manifest_sha256=sha256(base),samples=[row],
                          frames=[dict(frame,holdout_hamming=8)]))
    write_json(review,dict(queue_sha256=sha256(queue),admit=[0],exclude=[]))
    with pytest.raises(ValueError,match="Invalid external"):
        admitted(tmp_path,review,base)


@pytest.mark.parametrize("changes,valid", [({}, True),
    ({"source_image": "train/IMG_5123.jpg"}, False),
    ({"label": "red_balloon"}, False), ({"annotation_id": 99}, False),
    ({"box": [1, 1, 3, 3]}, False), ({"crop_box": [-1, 0, 30, 30]}, False),
    ({"anchor_id": 7}, False)])
def test_proposal_views_require_exact_reviewed_training_anchor(tmp_path, monkeypatch, changes, valid):
    from scripts import build_indoor_balloon_training as builder
    parent = tmp_path / "parent"
    proposal = tmp_path / "proposal"
    parent.mkdir()
    proposal.mkdir()
    write_json(parent / "review.json", {})
    write_json(parent / "decision.json", {})
    write_json(proposal / "review.json", dict(
        parent_queue_sha256=sha256(parent / "review.json"),
        parent_review_sha256=sha256(parent / "decision.json")))
    anchor = dict(id=0, label="blue_balloon", source_image="train/frame_0012.jpg",
                  source_sha256="pixels", session="train-session", annotation_id=1, frame_id=0)
    row = dict(anchor, anchor_id=0, box=[10, 10, 30, 30], crop_box=[8, 8, 32, 32],
               scan_box=[10, 10, 30, 30])
    row.update(changes)
    monkeypatch.setattr(builder, "admitted", lambda root, *args: [anchor] if root == parent else [row])
    if valid:
        assert builder.checked_proposal_views(proposal, "unused", "unused", parent,
                                              parent / "decision.json") == [row]
    else:
        with pytest.raises(ValueError, match="Invalid proposal"):
            builder.checked_proposal_views(proposal, "unused", "unused", parent, parent / "decision.json")
