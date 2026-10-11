import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from dtr.data import read_json, sha256, write_json
from scripts import public_balloon_training as public


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    queue = tmp_path/"queue"
    queue.mkdir()
    rgb = np.random.default_rng(42).integers(0,256,(240,320,3),dtype=np.uint8)
    Image.fromarray(rgb).save(queue/"frame.png")
    Image.fromarray(rgb[30:80,20:70]).save(queue/"crop.png")
    source = "train/IMG_20250714_181545_jpg.rf.abc.jpg"
    raw = io.BytesIO()
    Image.fromarray(rgb).save(raw,format="PNG")
    archive = tmp_path/"source.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr(source,raw.getvalue())
        z.writestr("train/_annotations.coco.json",json.dumps(dict(images=[dict(
            id=1,file_name=source.removeprefix("train/"),width=320,height=240)])))
    archive_sha = sha256(archive)
    f = dict(id=0,source=source,filename_family="astek-date-20250714",reference_hamming=20,
             selected_hamming=None,upstream_holdout_hamming=None,path="frame.png",native_path="frame.png",
             sha256=sha256(queue/"frame.png"),native_sha256=sha256(queue/"frame.png"),
             source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),source_group="group")
    r = dict(id=0,frame_id=0,path="crop.png",sha256=sha256(queue/"crop.png"),split="train",
             source_image=source,source_group="group",crop_box=[20,30,70,80],box=[25,35,65,75])
    write_json(queue/"review.json",dict(archive_sha256=archive_sha,source_profile="astek-balloons",
               deployment_approved=False,reference_inputs={},frames=[f],samples=[r]))
    review = tmp_path/"review.json"
    write_json(review,dict(queue_sha256=sha256(queue/"review.json"),archive_sha256=archive_sha,
               training_approved=True,split="train",source_namespace="astek-balloons-v2",deployment_approved=False,
               admit=[0],exclude=[],labels={"0":"blue_balloon"}))
    monkeypatch.setattr(public,"ASTEK_ARCHIVE_SHA",archive_sha)
    monkeypatch.setattr(public,"QUEUE_SHA",sha256(queue/"review.json"))
    monkeypatch.setattr(public,"REVIEW_SHA",sha256(review))
    return queue,review,archive


def test_source_crop_integrity_and_training_namespace(inputs):
    rows,frames = public.checked_rows(*inputs)
    assert len(rows)==len(frames)==1
    assert rows[0]["label"]=="blue_balloon"
    assert rows[0]["session"]=="astek-balloons-v2:astek-date-20250714"
    assert rows[0]["source_namespace"]=="astek-balloons-v2"


@pytest.mark.parametrize("which",["archive","queue","review","crop","frame"])
def test_changed_files_are_rejected(inputs,which):
    queue,review,archive = inputs
    path = {"archive":archive,"queue":queue/"review.json","review":review,
            "crop":queue/"crop.png","frame":queue/"frame.png"}[which]
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="changed"):
        public.checked_rows(*inputs)


def test_review_must_be_complete_even_if_hashes_rebound(inputs,monkeypatch):
    _,review,_ = inputs
    d=read_json(review)
    d["exclude"]=[0]
    write_json(review,d)
    monkeypatch.setattr(public,"REVIEW_SHA",sha256(review))
    with pytest.raises(ValueError,match="review"):
        public.checked_rows(*inputs)


def test_bad_crop_pixels_rejected_even_when_recorded_hash_updated(inputs,monkeypatch):
    queue,review,_ = inputs
    Image.new("RGB",(50,50),"blue").save(queue/"crop.png")
    q=read_json(queue/"review.json")
    q["samples"][0]["sha256"]=sha256(queue/"crop.png")
    write_json(queue/"review.json",q)
    monkeypatch.setattr(public,"QUEUE_SHA",sha256(queue/"review.json"))
    r=read_json(review)
    r["queue_sha256"]=public.QUEUE_SHA
    write_json(review,r)
    monkeypatch.setattr(public,"REVIEW_SHA",sha256(review))
    with pytest.raises(ValueError,match="Crop differs"):
        public.checked_rows(*inputs)


def test_matched_build_preserves_holdout_and_never_opens_test(inputs,tmp_path):
    base=tmp_path/"base"
    base.mkdir()
    rows=[]
    for split in ("train","val","test"):
        for j,label in enumerate(public.LABELS):
            name=f"{split}-{label}.png"
            if split=="test":
                digest=hashlib.sha256(name.encode()).hexdigest()
            else:
                Image.new("RGB",(10,10),(j*50,len(rows)*20,1)).save(base/name)
                digest=sha256(base/name)
            rows.append(dict(path=name,sha256=digest,session=f"base-{split}",split=split,
                             label=label,source_image=f"original/{name}"))
    write_json(base/"manifest.json",dict(samples=rows,task="balloon",classes=list(public.LABELS),
                                        training_approved=True,synthetic=False))
    outputs=[]
    for mode in ("control","admit"):
        out=tmp_path/mode
        receipt=public.build(base/"manifest.json",*inputs,out,mode,repetitions=2)
        doc=read_json(out/"manifest.json")
        assert receipt["added_entries"]==2
        assert receipt["unique_added_crops"]==(1 if mode=="admit" else 0)
        assert not doc["deployment_approved"] and not doc["test_pixels_materialized"]
        assert all(not (out/r["path"]).exists() for r in doc["samples"] if r["split"]=="test")
        assert [dict(r,path=r["path"].removeprefix("base/")) for r in doc["samples"][:9]]==rows
        outputs.append(receipt["counts"])
        with pytest.raises(FileExistsError):
            public.build(base/"manifest.json",*inputs,out,mode)
    assert outputs[0]==outputs[1]


@pytest.mark.parametrize("count",[0,5,True])
def test_repetition_bound_precedes_io(count):
    with pytest.raises(ValueError):
        public.build("missing","missing","missing","missing","missing","admit",count)
