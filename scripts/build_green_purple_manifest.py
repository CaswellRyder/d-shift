"""Green/purple balloon training manifest with red/blue balloons as hard negatives.

The capture targets are green and purple balloons; red and blue belong to the
drones (user correction, 2026-10-12). This copies the reviewed DTR v10 green/purple
crops unchanged and adds EngDes2 red/blue balloon crops labeled ``background`` so
the student learns to reject red/blue. The EngDes2 ``IMG_`` family and upstream
test split are never read (AGENTS.md dataset boundary).
"""
import argparse
import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

RED_BLUE = {"red_ballon": "red", "Kirmizi Balon": "red", "blue_ballon": "blue",
            "blue_baloon": "blue"}
FRAME_BLOCK = 200  # frame_#### numbers per temporal session
FRAME_GUARD = 10  # frames dropped next to each block edge to limit near-duplicates


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def source_group(name):
    """Upstream augmentations of one photo share the name before ``_ext.rf.``."""
    return re.sub(r"_(jpe?g|png)\.rf\..*$", "", name, flags=re.IGNORECASE)


def session_for(group):
    """Frame sequences split into temporal blocks; other photos by filename family."""
    match = re.fullmatch(r"frame_(\d+)", group)
    if match:
        number = int(match.group(1))
        offset = number % FRAME_BLOCK
        if offset < FRAME_GUARD or offset >= FRAME_BLOCK - FRAME_GUARD:
            return None
        return f"engdes2:frame-block-{number // FRAME_BLOCK}"
    return "engdes2:family-" + re.sub(r"\d+", "#", group)


def session_splits(candidates, val_fraction):
    """Whole sessions in hash order go to val until it holds ``val_fraction`` of crops."""
    counts = Counter(session for session, _, _ in candidates)
    splits, in_val = {}, 0
    for session in sorted(counts, key=lambda s: hashlib.sha256(s.encode()).hexdigest()):
        if in_val < val_fraction * len(candidates):
            splits[session], in_val = "val", in_val + counts[session]
        else:
            splits[session] = "train"
    return splits


def red_blue_negatives(archive, output, caps, val_fraction, seed):
    rng = random.Random(seed)
    rows, skipped = [], Counter()
    with zipfile.ZipFile(archive) as z:
        candidates = []
        for upstream in ("train", "valid"):
            doc = json.loads(z.read(f"{upstream}/_annotations.coco.json"))
            categories = {c["id"]: c["name"] for c in doc["categories"]}
            images = {i["id"]: i for i in doc["images"]}
            by_group = defaultdict(list)
            for ann in doc["annotations"]:
                info = images[ann["image_id"]]
                name = info["file_name"]
                if name.startswith("IMG_"):
                    skipped["reserved IMG_ family"] += 1
                    continue
                color = RED_BLUE.get(categories[ann["category_id"]])
                if color is None:
                    skipped[f"category {categories[ann['category_id']]}"] += 1
                    continue
                if min(ann["bbox"][2:]) < 8:
                    skipped["box under 8 px"] += 1
                    continue
                by_group[source_group(name)].append((upstream, info, ann, color))
            for group, anns in sorted(by_group.items()):
                session = session_for(group)
                if session is None:
                    skipped["frame block guard"] += len(anns)
                    continue
                # One crop per source photo: augmentations are near-duplicates.
                candidates.append((session, group, rng.choice(anns)))
        rng.shuffle(candidates)
        splits = session_splits(candidates, val_fraction)
        taken, seen_digests, seen_groups = Counter(), set(), set()
        for session, group, (upstream, info, ann, color) in candidates:
            split = splits[session]
            if taken[split] >= caps[split] or group in seen_groups:
                continue
            raw = z.read(f"{upstream}/{info['file_name']}")
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            width, height = image.size
            x, y, w, h = ann["bbox"]
            pad = 0.12 * max(w, h)  # Same padding as prepare_coco and the search.
            box = (max(0, int(x - pad)), max(0, int(y - pad)),
                   min(width, int(x + w + pad)), min(height, int(y + h + pad)))
            source_digest = sha256_bytes(raw)
            rel = f"images/{split}/engdes2-{color}-{source_digest[:20]}.png"
            dest = output / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.crop(box).save(dest)
            digest = sha256_bytes(dest.read_bytes())
            if digest in seen_digests:
                dest.unlink()
                continue
            seen_digests.add(digest)
            seen_groups.add(group)
            taken[split] += 1
            rows.append(dict(
                path=rel, split=split, session=session, label="background", sha256=digest,
                source_image=f"{upstream}/{info['file_name']}", source_sha256=source_digest,
                source_group=group, crop_box=list(box), annotation_id=ann["id"],
                negative_kind=f"{color}_balloon",
                label_origin="upstream EngDes2 red/blue box; red/blue are non-target drone colors",
            ))
    return rows, taken, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="data/dtr-proposals-reviewed-20261005/balloon")
    parser.add_argument("--engdes2", default="data/raw/engdes2-red-blue-v1-20261008/dataset.zip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-negatives", type=int, default=1800)
    parser.add_argument("--val-negatives", type=int, default=300)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    base, output = Path(args.base), Path(args.output)
    if output.exists():
        parser.error(f"{output} exists; choose a new directory")
    family = json.loads(Path("configs/engdes2-development-family.json").read_text())
    with open(args.engdes2, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != family["source_archive_sha256"]:
            parser.error("EngDes2 archive hash does not match the recorded export")
    doc = json.loads((base / "manifest.json").read_text())
    if doc["classes"] != ["background", "green_balloon", "purple_balloon"]:
        parser.error("Base manifest is not the green/purple task")
    output.mkdir(parents=True)
    for entry in base.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, output / entry.name)
    negatives, taken, skipped = red_blue_negatives(
        Path(args.engdes2), output,
        dict(train=args.train_negatives, val=args.val_negatives),
        args.val_fraction, args.seed)
    samples = doc["samples"] + negatives
    manifest = dict(
        doc,
        samples=samples,
        source=dict(base=str(base / "manifest.json"),
                    base_sha256=sha256_bytes((base / "manifest.json").read_bytes()),
                    hard_negatives=args.engdes2,
                    hard_negatives_sha256=family["source_archive_sha256"]),
        hard_negative_policy=dict(
            reason="Capture targets are green/purple; red/blue are drone colors (2026-10-12)",
            label="background", taken=dict(taken), skipped=dict(skipped),
            excluded=["EngDes2 IMG_ family (reserved development evaluation)",
                      "EngDes2 upstream test split (reserved)",
                      "EngDes2 'balloon' and 'NEWEST' categories (color unknown)"],
            sessions="frame_ blocks of 200 with 10-frame edge guards; other photos by family",
            seed=args.seed),
        training_approved=True,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(
        counts=Counter(f"{r['split']}/{r['label']}" for r in samples),
        negatives=dict(taken), skipped=dict(skipped)), indent=2))


if __name__ == "__main__":
    main()
