"""Quarantine added split candidates from admitted TRAIN sources; no auto-labels."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, validate, write_json
from dtr.tracking import iou
from scripts.audit_balloon_search_stages import original_frames, indoor_frames, training_source
from scripts.research_balloon_search import experimental


def unmatched_acceptances(rgb, candidates, truth, predictor):
    """Find suspicious TRAIN predictions for review, never assign negative labels.

    Missing annotations are not evidence of background. Store the prediction as
    provenance only; admitted/excluded labels must still come from visual review.
    """
    rows = []
    for candidate in candidates:
        if any(iou(candidate["box"], t["box"]) >= .1 for t in truth):
            continue
        x, y, z, w = candidate["crop_box"]
        prediction = predictor.predict(rgb[y:w, x:z])
        if prediction["accepted"] and prediction["label"] in ("red_balloon", "blue_balloon"):
            rows.append(dict(candidate, mining_prediction={
                k: prediction[k] for k in ("label", "score", "scores", "accepted")}))
    return rows


def prepare(manifest, original_manifest, original_archive, queue, review, indoor_archive, output,
            indoor_only=False, candidate_scope="added", variant="mser_instance_split", mining_model=None):
    if candidate_scope not in ("added", "all"):
        raise ValueError("Unknown candidate scope")
    if variant not in ("mser_instance_split", "mser_blue_cores"):
        raise ValueError("Unknown training proposal variant")
    if mining_model and candidate_scope != "all":
        raise ValueError("Mining requires all candidate proposals")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = read_json("configs/balloon-red-blue.json")
    base = validate(manifest, config, splits=("train", "val"))
    if base.get("training_approved") is not True or base.get("synthetic", True):
        raise ValueError("Require approved real base data")
    allowed = {r["source_image"]: r for r in base["samples"] if r["split"] == "train"}
    predictor, mining = None, None
    if mining_model:
        from scripts.research_predictor import load_predictor
        from dtr.vision import RED_BLUE_PROFILE, validate_model_profile
        predictor = load_predictor(mining_model)
        validate_model_profile(predictor.metadata, RED_BLUE_PROFILE)
        mining = dict(model_sha256=sha256(mining_model),
                      metadata_sha256=sha256(Path(mining_model).with_suffix(".json")),
                      maximum_annotation_iou_exclusive=.1,
                      threshold=predictor.metadata["threshold"], labels_automatic=False)
    panels = {} if indoor_only else {"original": original_frames(original_manifest, original_archive)}
    panels["indoor"] = indoor_frames(queue, review, original_manifest, indoor_archive)
    output.mkdir(parents=True)
    (output / "crops").mkdir()
    (output / "frames").mkdir()
    frames, rows, seen = [], [], set()
    for panel, sources in panels.items():
        for source, rgb, truth in sources:
            source = "train/"+source if panel == "indoor" else source
            training_source(source)
            if source not in allowed:
                raise ValueError("Source not in current training manifest")
            parent = allowed[source]
            baseline = experimental(rgb, "mser")
            additions = experimental(rgb, variant)
            if candidate_scope == "added":
                additions = additions[len(baseline):]
            if predictor is not None:
                additions = unmatched_acceptances(rgb, additions, truth, predictor)
            if not additions:
                continue
            frame_id = len(frames)
            relative = f"frames/{frame_id:03d}.png"
            Image.fromarray(rgb).save(output / relative)
            frames.append(dict(id=frame_id, path=relative, sha256=sha256(output / relative),
                               source_image=source, truth=truth, panel=panel))
            for candidate in additions:
                box = candidate["crop_box"]
                key = source, tuple(box)
                if key in seen:
                    continue
                seen.add(key)
                x, y, z, w = box
                crop_path = f"crops/{len(rows):03d}.png"
                Image.fromarray(rgb[y:w, x:z]).save(output / crop_path)
                rows.append(dict(id=len(rows), path=crop_path, sha256=sha256(output / crop_path),
                                 split="train", session=parent["session"], source_image=source,
                                 source_sha256=parent["source_sha256"], frame_id=frame_id,
                                 label="unreviewed", **candidate,
                                 overlaps=[dict(label=t["label"], iou=iou(candidate["box"], t["box"]))
                                           for t in truth]))
    doc = dict(samples=rows, frames=frames, base_manifest_sha256=sha256(manifest),
               indoor_only=indoor_only, candidate_scope=candidate_scope, proposal_variant=variant, mining=mining,
               inputs={"original_manifest": sha256(original_manifest), "original_archive": sha256(original_archive),
                       "queue": sha256(Path(queue)/"review.json"), "review": sha256(review),
                       "indoor_archive": sha256(indoor_archive)},
               script_sha256=sha256(__file__), search_sha256=sha256("scripts/research_balloon_search.py"),
               training_approved=False, deployment_approved=False, review_required=True,
               scope="Candidate crops of already admitted training scenes; incomplete scene labels; not independent photos")
    write_json(output / "review.json", doc)
    for start in range(0, len(rows), 24):
        sheet = Image.new("RGB", (960, 640), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(rows[start:start+24]):
            x, y = n % 6*160, n//6*160
            with Image.open(output / row["path"]) as im:
                sheet.paste(ImageOps.contain(im, (156, 130)), (x, y+25))
            draw.text((x+2, y+2), f"id {row['id']} / frame {row['frame_id']}", fill="black")
        sheet.save(output / f"crops-{start:03d}.png")
    for start in range(0, len(frames), 9):
        sheet = Image.new("RGB", (960, 810), "#ddd")
        draw = ImageDraw.Draw(sheet)
        for n, frame in enumerate(frames[start:start+9]):
            x, y = n % 3*320, n//3*270
            with Image.open(output / frame["path"]) as im:
                sheet.paste(im, (x, y+25))
            draw.text((x, y), f"frame {frame['id']}", fill="black")
        sheet.save(output / f"frames-{start:03d}.png")
    return dict(crops=len(rows), frames=len(frames), queue_sha256=sha256(output / "review.json"))


def checked_rows(manifest, root, decision_path):
    root = Path(root).resolve()
    queue, decision = read_json(root / "review.json"), read_json(decision_path)
    base = read_json(manifest)
    if (queue["base_manifest_sha256"] != sha256(manifest)
            or decision["queue_sha256"] != sha256(root / "review.json")
            or decision.get("training_approved") is not True
            or decision.get("deployment_approved") is not False):
        raise ValueError("Require matching explicit training review")
    labels = decision["labels"]
    excluded = decision["exclude"]
    ids = [r["id"] for r in labels] + excluded
    if (not labels or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)
            or set(ids) != set(range(len(queue["samples"])))):
        raise ValueError("Review must decide every crop once")
    allowed = {(r["source_image"], r["session"], r["source_sha256"]) for r in base["samples"] if r["split"] == "train"}
    forbidden = {r["sha256"] for r in base["samples"] if r["split"] != "train"}
    holdout_sources = {r["source_sha256"] for r in base["samples"] if r["split"] != "train"}
    output = []
    for label in labels:
        r = queue["samples"][label["id"]]
        training_source(r["source_image"])
        path = (root / r["path"]).resolve()
        if (r["id"] != label["id"] or r["split"] != "train" or label["label"] not in base["classes"]
                or (r["source_image"], r["session"], r["source_sha256"]) not in allowed
                or r["source_sha256"] in holdout_sources or r["sha256"] in forbidden
                or not path.is_relative_to(root) or sha256(path) != r["sha256"]):
            raise ValueError("Changed crop, unsafe source or invalid label")
        frame = queue["frames"][r["frame_id"]]
        frame_path = (root / frame["path"]).resolve()
        if (frame["source_image"] != r["source_image"] or not frame_path.is_relative_to(root)
                or sha256(frame_path) != frame["sha256"]):
            raise ValueError("Changed parent frame")
        x, y, z, w = r["crop_box"]
        if not all(type(v) is int for v in (x, y, z, w)) or not (0 <= x < z <= 320 and 0 <= y < w <= 240):
            raise ValueError("Invalid crop bounds")
        with Image.open(frame_path) as im, Image.open(path) as crop:
            if not np.array_equal(np.asarray(im.convert("RGB"))[y:w, x:z], np.asarray(crop.convert("RGB"))):
                raise ValueError("Crop does not match reviewed source geometry")
        output.append(dict(r, label=label["label"], label_origin="Explicit AI visual training-only review"))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "original-manifest", "original-archive", "queue", "review", "indoor-archive", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--indoor-only", action="store_true", help="Skip original sources when expanding an existing queue")
    p.add_argument("--candidate-scope", choices=("added", "all"), default="added",
                   help="Review only split additions or all candidates for proposal-domain training")
    p.add_argument("--variant", choices=("mser_instance_split", "mser_blue_cores"),
                   default="mser_instance_split", help="Training-only proposal generator; no automatic labels")
    p.add_argument("--mining-model", help="Review only accepted proposals with <0.1 IoU to annotated TRAIN targets; requires --candidate-scope all")
    a = p.parse_args()
    print(prepare(a.manifest, a.original_manifest, a.original_archive, a.queue, a.review, a.indoor_archive, a.output,
                  a.indoor_only, a.candidate_scope, a.variant, a.mining_model))
