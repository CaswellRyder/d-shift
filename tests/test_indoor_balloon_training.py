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
    dict(session="independent-holdout"),dict(sha256="reserved"),dict(path="../escape.png")])
def test_admission_rejects_unscoped_crops(tmp_path,changes):
    base,row,frame,queue,review = queue_fixture(tmp_path)
    write_json(queue,dict(base_manifest_sha256=sha256(base),samples=[dict(row,**changes)],frames=[frame]))
    write_json(review,dict(queue_sha256=sha256(queue),admit=[0],exclude=[]))
    with pytest.raises(ValueError,match="Invalid external"):
        admitted(tmp_path,review,base)


def test_near_holdout_is_not_training_material(tmp_path):
    base,row,frame,queue,review = queue_fixture(tmp_path)
    write_json(queue,dict(base_manifest_sha256=sha256(base),samples=[row],
                          frames=[dict(frame,holdout_hamming=8)]))
    write_json(review,dict(queue_sha256=sha256(queue),admit=[0],exclude=[]))
    with pytest.raises(ValueError,match="Invalid external"):
        admitted(tmp_path,review,base)
