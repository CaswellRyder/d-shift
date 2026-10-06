"""Render cached goal-detector development errors. Never generate training labels."""

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, write_json
from dtr.detector_metrics import match
from dtr.tracking import iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = read_json(args.report)
    if report["split"] != "val" or report["metrics"]["test_evaluated"]:
        raise ValueError("Development-validation diagnostics only")
    source = Path("data/roboflow-dtr-v10-grouped/coco/valid")
    if sha256(source / "_annotations.coco.json") != report["annotation_sha256"]:
        raise ValueError("Source annotations changed")
    groups = defaultdict(list)
    for frame in report["frames"]:
        truth = frame["truth"]
        predictions = [p for p in frame["predictions"]
                       if p["score"] >= report["standard"]["confidence_threshold"]]
        hits = match(truth, predictions, report["standard"]["iou_threshold"])
        used = {target for _, target in hits if target is not None}
        for pred, target in hits:
            if target is not None:
                continue
            nearest = max(truth, key=lambda t: iou(pred["box"], t["box"]), default=None)
            groups[pred["label"] + "-unmatched"].append(dict(
                file=frame["file"], subject=pred, truth=truth,
                nearest_label=nearest["label"] if nearest else "none",
                nearest_iou=iou(pred["box"], nearest["box"]) if nearest else 0,
            ))
        for index, target in enumerate(truth):
            if index in used:
                continue
            nearest = max(predictions, key=lambda p: iou(target["box"], p["box"]), default=None)
            groups[target["label"] + "-missed"].append(dict(
                file=frame["file"], subject=target, truth=truth,
                nearest_label=nearest["label"] if nearest else "none",
                nearest_iou=iou(target["box"], nearest["box"]) if nearest else 0,
            ))
    output.mkdir(parents=True)
    receipt = dict(report_sha256=sha256(args.report), training_approved=False,
                   label_corrections=False, test_evaluated=False, groups={},
                   scope="Cached errors at fixed operating point; not prevalence or corrected labels")
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: -r["subject"].get("score", 0))
        chosen, seen = [], set()
        for row in ordered:
            if row["file"] not in seen:
                seen.add(row["file"])
                chosen.append(row)
            if len(chosen) == 16:
                break
        sheet = Image.new("RGB", (1120, 720), "white")
        pen = ImageDraw.Draw(sheet)
        for index, row in enumerate(chosen):
            with Image.open(source / row["file"]) as opened:
                original = opened.convert("RGB")
            marked = original.copy()
            draw = ImageDraw.Draw(marked)
            for t in row["truth"]:
                draw.rectangle(t["box"], outline="cyan", width=2)
            box = row["subject"]["box"]
            draw.rectangle(box, outline="red", width=2)
            pad = max(5, round(max(box[2]-box[0], box[3]-box[1])*0.15))
            crop = original.crop((max(0, int(box[0])-pad), max(0, int(box[1])-pad),
                                  min(original.width, int(box[2])+pad),
                                  min(original.height, int(box[3])+pad)))
            x, y = index % 4 * 280, index // 4 * 180
            sheet.paste(ImageOps.contain(marked, (136, 130)), (x, y))
            sheet.paste(ImageOps.contain(crop, (136, 130)), (x+140, y))
            pen.text((x, y+132), f"{index}: {row['subject']['label']}", fill="black")
            pen.text((x, y+145), f"nearest {row['nearest_label']}", fill="black")
            pen.text((x, y+158), f"IoU {row['nearest_iou']:.2f}; "
                     f"score {row['subject'].get('score', 'GT')}", fill="black")
        sheet.save(output / f"{key}.jpg")
        write_json(output / f"{key}.json", chosen)
        receipt["groups"][key] = dict(total=len(rows), shown=len(chosen))
    write_json(output / "review.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
