import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from dtr.data import read_json, sha256, write_json
from scripts import prepare_blue_source_review as review
from scripts import prepare_indoor_balloon_review as indoor


def test_source_family_is_only_a_conservative_filename_hint():
    assert review.family("left0001_jpg.rf.abc.jpg") == "left-sequence"
    assert review.family("blue-12_png_jpg.rf.abc.jpg") == "blue-sequence"
    assert review.family("20211201.rf.abc.jpg") == "other-unverified"
    with pytest.raises(ValueError, match="Reserved"):
        review.family("IMG_123.rf.abc.jpg")


def test_scan_geometry_and_padding_keep_processing_coordinates():
    assert review.scan_box([100,100,100,200],640,640) == [50,37.5,100,112.5]
    assert review.crop_box([0,0,320,240]) == [0,0,320,240]


@pytest.mark.parametrize("box", [[0,0,0,10],[-1,0,2,3],[0,0,641,3],[0,0,np.nan,3],[1,2,3]])
def test_bad_source_boxes_rejected(box):
    with pytest.raises(ValueError):
        review.scan_box(box,640,640)


@pytest.fixture
def dataset(tmp_path,monkeypatch):
    root = tmp_path/"raw"
    root.mkdir()
    archive = root/"dataset.zip"
    names = ["blue-1_png.rf.aaa.png","left0001_jpg.rf.bbb.png",
             "IMG_forbidden.rf.ccc.png","left0001_jpg.rf.ddd.png"]
    images = [dict(id=i,file_name=n,width=320,height=240) for i,n in enumerate(names)]
    held = dict(id=99,file_name="blue-1_png.rf.eee.png",width=320,height=240)
    classes = [dict(id=1,name="FRC 2022 - v1 ball")]
    rng = np.random.default_rng(711)
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("train/_annotations.coco.json",json.dumps(dict(images=images,categories=classes,
            annotations=[dict(id=i,image_id=i,category_id=1,bbox=[50,50,30,40]) for i in range(4)])))
        z.writestr("valid/_annotations.coco.json",json.dumps(dict(images=[held],categories=classes,annotations=[])))
        # Reserved IMG pixels are deliberately absent: they must never be read.
        for split,rows in (("train",[images[0],images[1],images[3]]),("valid",[held])):
            for r in rows:
                out = io.BytesIO()
                Image.fromarray(rng.integers(0,256,(240,320,3),dtype=np.uint8)).save(out,format="PNG")
                z.writestr(split+"/"+r["file_name"],out.getvalue())
    write_json(root/"download-receipt.json",dict(archive_sha256=sha256(archive),source="https://example.org/source"))
    monkeypatch.setattr(review,"ARCHIVE_SHA",sha256(archive))
    monkeypatch.setattr(review,"references",lambda:([rng.integers(0,2,63).astype(bool)],{"fixture":"hash"}))
    return root


def test_quarantine_never_admits_labels_or_cross_split_sources(dataset,tmp_path):
    output = tmp_path/"queue"
    result = review.prepare(dataset,output,1)
    doc = read_json(output/"review.json")
    assert result["frames"] == 1
    assert doc["frames"][0]["source_group"] == "left0001_jpg.png"
    assert doc["training_approved"] is False and doc["label_mapping_approved"] is False
    assert doc["test_evaluated"] is False and doc["heldout_pixels_used_for_duplicate_screening_only"]
    assert all(r["label"] == "unreviewed" for r in doc["samples"])
    assert sum(r["kind"] == "upstream_object" for r in doc["samples"]) == 1
    for r in doc["samples"]:
        a,b,c,d = r["crop_box"]
        f = doc["frames"][r["frame_id"]]
        with Image.open(output/f["path"]) as im:
            pixels = np.asarray(im)
        with Image.open(output/r["path"]) as im:
            assert np.array_equal(np.asarray(im),pixels[b:d,a:c])
    with pytest.raises(FileExistsError):
        review.prepare(dataset,output)


def test_source_hash_change_is_rejected_before_output(dataset,tmp_path):
    write_json(dataset/"download-receipt.json",dict(archive_sha256="wrong"))
    output = tmp_path/"queue"
    with pytest.raises(ValueError, match="archive changed"):
        review.prepare(dataset,output)
    assert not output.exists()


@pytest.mark.parametrize("count", [0,13,True])
def test_invalid_review_budget_rejected_without_io(count):
    with pytest.raises(ValueError, match="1..12"):
        review.prepare("missing","missing",count)


def test_vectorized_duplicate_distance_matches_original_all_reflections(monkeypatch):
    rng = np.random.default_rng(99)
    rgb = rng.integers(0,256,(120,160,3),dtype=np.uint8)
    refs = [rng.integers(0,2,63).astype(bool) for _ in range(41)]
    actual_fingerprint = indoor.fingerprint
    expected = min(int(np.count_nonzero(actual_fingerprint(view) != ref))
                   for view in (rgb,rgb[::-1],rgb[:,::-1],rgb[::-1,::-1]) for ref in refs)
    calls = []

    def counted(view):
        calls.append(1)
        return actual_fingerprint(view)

    monkeypatch.setattr(indoor,"fingerprint",counted)
    assert indoor.holdout_distance(rgb,refs) == expected
    assert len(calls) == 4
    for view in (rgb,rgb[::-1],rgb[:,::-1],rgb[::-1,::-1]):
        assert indoor.holdout_distance(rgb,[actual_fingerprint(view)]) == 0
    with pytest.raises(ValueError):
        indoor.holdout_distance(rgb,[])
