"""Hash-bound Astek training admission, matched controls and target diagnostics.

The reviewed set is training-only and has incomplete full-frame truth. Target
coverage is not scene precision/recall or independent qualification.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, validate, write_json
from scripts.prepare_blue_source_review import ASTEK_ARCHIVE_SHA, astek_family, decoded
from scripts.audit_public_balloon_export import bounded_read

QUEUE_SHA = "bd507a0b143c5495e4ecb051a76cd5a2345fb72ad59b7929dac140ee4e26e1f6"
REVIEW_SHA = "3d643e71941c3ab2344a6e002d025534be1cb56f2d91c29e40a4cbe5da9c506c"
LABELS = ("background", "red_balloon", "blue_balloon")


def checked_image(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha256(path) != expected:
        raise ValueError("Image escaped root or changed")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def checked_rows(queue_root, review_path, archive_path):
    import json
    root = Path(queue_root).resolve()
    if (sha256(root/"review.json") != QUEUE_SHA or sha256(review_path) != REVIEW_SHA
            or sha256(archive_path) != ASTEK_ARCHIVE_SHA):
        raise ValueError("Approved source, queue or review changed")
    queue, review = read_json(root/"review.json"), read_json(review_path)
    if (review.get("queue_sha256") != QUEUE_SHA or queue.get("archive_sha256") != ASTEK_ARCHIVE_SHA
            or review.get("archive_sha256") != ASTEK_ARCHIVE_SHA
            or review.get("training_approved") is not True or review.get("split") != "train"
            or review.get("source_namespace") != "astek-balloons-v2"
            or queue.get("source_profile") != "astek-balloons"
            or queue.get("deployment_approved") is not False
            or review.get("deployment_approved") is not False):
        raise ValueError("Require explicit training-only source identity")
    ids = review["admit"] + review["exclude"]
    if (not review["admit"] or any(type(i) is not int for i in ids)
            or len(ids) != len(set(ids)) or set(ids) != set(range(len(queue["samples"])))
            or set(review["labels"]) != {str(i) for i in review["admit"]}
            or any(v not in LABELS for v in review["labels"].values())):
        raise ValueError("Incomplete or invalid label review")
    for path, digest in queue["reference_inputs"].items():
        if sha256(path) != digest:
            raise ValueError("Duplicate-screen reference changed")
    frame_pixels = {}
    with zipfile.ZipFile(archive_path) as archive:
        doc = json.loads(bounded_read(archive,"train/_annotations.coco.json",50_000_000))
        sources = {"train/"+r["file_name"]:r for r in doc["images"]}
        for f in queue["frames"]:
            fam = astek_family(Path(f["source"]).name)
            if (f["id"] in frame_pixels or fam is None or f["filename_family"] != fam
                    or f["reference_hamming"] <= 8
                    or (f["selected_hamming"] is not None and f["selected_hamming"] <= 8)
                    or f["upstream_holdout_hamming"] is not None):
                raise ValueError("Invalid training frame or duplicate screen")
            original = decoded(archive,f["source"],sources[f["source"]])
            native = checked_image(root,f["native_path"],f["native_sha256"])
            scan = checked_image(root,f["path"],f["sha256"])
            if (hashlib.sha256(original.tobytes()).hexdigest() != f["source_sha256"]
                    or not np.array_equal(original,native)
                    or not np.array_equal(cv2.resize(original,(320,240),interpolation=cv2.INTER_AREA),scan)):
                raise ValueError("Frame differs from pinned original")
            frame_pixels[f["id"]] = scan
    frames = {f["id"]:f for f in queue["frames"]}
    rows = []
    for i in review["admit"]:
        r = queue["samples"][i]
        f = frames[r["frame_id"]]
        box = r["crop_box"]
        if (r["id"] != i or r["split"] != "train" or r["source_image"] != f["source"]
                or r["source_group"] != f["source_group"] or len(box) != 4
                or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= 320 and 0 <= box[1] < box[3] <= 240)):
            raise ValueError("Invalid reviewed crop identity or bounds")
        a,b,c,d = box
        crop = checked_image(root,r["path"],r["sha256"])
        if not np.array_equal(crop,frame_pixels[f["id"]][b:d,a:c]):
            raise ValueError("Crop differs from original frame region")
        rows.append(dict(r,label=review["labels"][str(i)],source_sha256=f["source_sha256"],
                         source_archive_sha256=ASTEK_ARCHIVE_SHA,source_namespace="astek-balloons-v2",
                         session="astek-balloons-v2:"+f["filename_family"],
                         label_origin="assistant visually reviewed crop and full-frame context"))
    return rows,frame_pixels


def build(manifest, queue, review, archive, output, mode, repetitions=4):
    if mode not in ("admit","control") or type(repetitions) is not int or not 1 <= repetitions <= 4:
        raise ValueError("Require admit/control and 1..4 repetitions")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = read_json("configs/balloon-red-blue.json")
    base = validate(manifest,config,splits=("train","val"))
    if base.get("synthetic",True) or base.get("training_approved") is not True:
        raise ValueError("Require approved real base")
    extra,_ = checked_rows(queue,review,archive)
    for row in base["samples"]:
        if row["split"] == "train" and ("instance_ablation_mode" in row
                or Path(row["source_image"]).name.lower().startswith("img_")):
            raise ValueError("Base already has public experiment or reserved IMG rows")
    for e in extra:
        for r in base["samples"]:
            same = e["sha256"] == r["sha256"] or e["source_sha256"] == r.get("source_sha256")
            if same and (r["split"] != "train" or e["label"] != r["label"]):
                raise ValueError("New row conflicts with an existing source or label")
    output.mkdir(parents=True)

    def copy(row, root, prefix):
        relative = f"{prefix}/{row['path']}"
        if row["split"] != "test":
            dest = output/relative
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(Path(root)/row["path"],dest)
        return dict(row,path=relative)

    samples = [copy(r,Path(manifest).parent,"base") for r in base["samples"]]
    pools = {label:[r for r in samples if r["split"] == "train" and r["label"] == label] for label in LABELS}
    rng = np.random.default_rng(47)
    for repetition in range(repetitions):
        for r in extra:
            selected = (copy(r,queue,"public") if mode == "admit"
                        else dict(pools[r["label"]][int(rng.integers(len(pools[r["label"]])))]))
            samples.append(dict(selected,instance_ablation_mode=mode,instance_repetition=repetition))
    doc = dict(base,samples=samples,parent_manifest_sha256=sha256(manifest),
               public_queue_sha256=QUEUE_SHA,public_review_sha256=REVIEW_SHA,
               public_source_archive_sha256=ASTEK_ARCHIVE_SHA,public_ablation_mode=mode,
               public_repetitions=repetitions,test_pixels_materialized=False,deployment_approved=False)
    write_json(output/"manifest.json",doc)
    validate(output/"manifest.json",config,splits=("train","val"))
    receipt = dict(mode=mode,manifest_sha256=sha256(output/"manifest.json"),
                   parent_manifest_sha256=sha256(manifest),script_sha256=sha256(__file__),
                   added_entries=len(extra)*repetitions,unique_added_crops=len(extra) if mode=="admit" else 0,
                   counts=dict(Counter(f"{r['split']}/{r['label']}" for r in samples)),
                   test_evaluated=False,deployment_approved=False)
    write_json(output/"receipt.json",receipt)
    return receipt


def audit(queue, review, archive, model):
    from dtr.runtime import Predictor
    from dtr.tracking import iou
    from scripts.pi_balloon_search_bench import process_frame
    rows,pixels = checked_rows(queue,review,archive)
    predictor = Predictor(model,allow_unvalidated=True)
    targets = []
    for ident,rgb in pixels.items():
        found = process_frame(rgb,predictor,"mser",selection="split_merged",split_max_aspect=2.42913421076193)
        for r in rows:
            if r["frame_id"] != ident or r["label"] == "background":
                continue
            detections = found["detections"]
            def best(ds):
                return max((iou(r["box"],d["box"]) for d in ds),default=0.)
            targets.append(dict(id=r["id"],frame_id=ident,label=r["label"],
                crop_prediction=predictor.predict(Path(queue)/r["path"])["label"],
                proposal_iou=best(detections),
                classified_iou=best(d for d in detections if d["raw_accepted"] and d["label"]==r["label"]),
                selected_iou=best(d for d in detections if d["accepted"] and d["label"]==r["label"])))
    return dict(model=str(model),model_sha256=sha256(model),queue_sha256=QUEUE_SHA,review_sha256=REVIEW_SHA,
                script_sha256=sha256(__file__),targets=targets,
                coverage={stage:dict(Counter(t["label"] for t in targets if t[stage] >= .5))
                          for stage in ("proposal_iou","classified_iou","selected_iou")},
                scope="Reviewed training target coverage only; incomplete scene truth means no full-scene precision or qualification claim",
                test_evaluated=False,deployment_approved=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("queue","review","archive","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--manifest")
    p.add_argument("--model")
    p.add_argument("--mode",required=True,choices=("admit","control","audit"))
    a = p.parse_args()
    if a.mode == "audit":
        if not a.model or Path(a.output).exists():
            p.error("Audit requires model and a new output path")
        result = audit(a.queue,a.review,a.archive,a.model)
        write_json(a.output,result)
        print(result["coverage"])
    else:
        if not a.manifest:
            p.error("Build requires manifest")
        print(build(a.manifest,a.queue,a.review,a.archive,a.output,a.mode))
