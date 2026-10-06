"""Dataset manifests, leakage checks, deterministic preprocessing, COCO import."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SPLITS = ("train", "val", "test")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rgb(path, size):
    # Exactly the same stretch-to-square bilinear policy in training and inference.
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32
        )


def validate(manifest, config, splits=SPLITS):
    if not splits or not set(splits) <= set(SPLITS):
        raise ValueError("Invalid validation splits")
    path = Path(manifest).resolve()
    doc = read_json(path)
    if doc["classes"] != config["classes"]:
        raise ValueError("Manifest class order does not match config")
    if doc.get("task") != config["task"]:
        raise ValueError("Manifest task does not match config")
    rows = doc["samples"]
    hashes, sessions = {}, {}
    counts = Counter()
    for row in rows:
        split, label = row["split"], row["label"]
        if split not in SPLITS or label not in doc["classes"]:
            raise ValueError(f"Invalid split/label: {row}")
        image = (path.parent / row["path"]).resolve()
        if not image.is_relative_to(path.parent):
            raise ValueError("Image path escapes dataset root")
        recorded_digest = row.get("sha256")
        if (not isinstance(recorded_digest, str) or len(recorded_digest) != 64
                or any(c not in "0123456789abcdef" for c in recorded_digest)):
            raise ValueError(f"Dataset image checksum changed or invalid: {image}")
        # Unselected splits retain metadata leakage checks without opening their images.
        digest = sha256(image) if split in splits else recorded_digest
        if row.get("sha256") != digest:
            raise ValueError(f"Dataset image changed: {image}")
        for seen, key, description in [
            (hashes, digest, "image"),
            (sessions, row["session"], "session"),
        ]:
            if key in seen and seen[key] != split:
                raise ValueError(f"Cross-split {description} leakage: {key}")
            seen[key] = split
        counts[(split, label)] += 1
    for split in splits:
        for label in doc["classes"]:
            if not counts[(split, label)]:
                raise ValueError(f"No examples for {split}/{label}")
    return doc


def batches(manifest, config, split, size, shuffle=False, augment=False, paired=False,
            validation_splits=SPLITS):
    import tensorflow as tf

    if split not in validation_splits:
        raise ValueError("Requested split is outside validated scope")
    doc = validate(manifest, config, splits=validation_splits)
    if split == "train" and doc.get("training_approved") is False:
        raise ValueError("Dataset is quarantined pending label review")
    root = Path(manifest).resolve().parent
    rows = [r for r in doc["samples"] if r["split"] == split]
    labels = doc["classes"]

    def generate():
        for row in rows:
            x = load_rgb(root / row["path"], size)
            if paired:
                x = {"teacher": x, "student": load_rgb(root / row["path"], config["student_size"])}
            yield x, np.int32(labels.index(row["label"]))

    signature = tf.TensorSpec((size, size, 3), tf.float32)
    if paired:
        signature = {
            "teacher": signature,
            "student": tf.TensorSpec(
                (config["student_size"], config["student_size"], 3), tf.float32
            ),
        }
    ds = tf.data.Dataset.from_generator(
        generate, output_signature=(signature, tf.TensorSpec((), tf.int32))
    )
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(rows)))
    if shuffle:
        ds = ds.shuffle(min(len(rows), 1024), seed=config["seed"])
    if augment:
        # No hue shifts: class identity depends on color.
        def transform(x, y):
            flip = tf.random.uniform(()) < 0.5
            brightness = tf.random.uniform((), -15, 15)

            def apply(image):
                image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
                return tf.clip_by_value(image + brightness, 0, 255)

            return ({k: apply(v) for k, v in x.items()} if paired else apply(x)), y

        ds = ds.map(transform, num_parallel_calls=1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.batch(config["batch_size"]).with_options(options).prefetch(1)


def synthetic(output, config, per_class=12):
    """Engineering fixture ONLY. Deliberately marked synthetic throughout artifacts."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(config["seed"])
    rows = []
    for split in SPLITS:
        for label in config["classes"]:
            for i in range(per_class):
                image = Image.fromarray(rng.integers(0, 65, (96, 96, 3), dtype=np.uint8))
                draw = ImageDraw.Draw(image)
                color = (70, 220, 60) if "green" in label else (170, 40, 220)
                if "orange" in label:
                    color = (250, 130, 15)
                elif "yellow" in label:
                    color = (245, 235, 25)
                a, b = rng.integers(8, 22, size=2)
                box = (int(a), int(b), 85, 85)
                if "balloon" in label:
                    draw.ellipse(box, fill=color)
                elif "circle" in label:
                    draw.ellipse(box, outline=color, width=7)
                elif "square" in label:
                    draw.rectangle(box, outline=color, width=7)
                elif "triangle" in label:
                    draw.line(
                        [(48, int(b)), (85, 85), (int(a), 85), (48, int(b))], fill=color, width=7
                    )
                rel = f"images/{split}/{label}/{i:04d}.png"
                dest = output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                image.save(dest)
                rows.append(
                    dict(
                        path=rel,
                        label=label,
                        split=split,
                        session=f"synthetic-{split}",
                        sha256=sha256(dest),
                    )
                )
    write_json(
        output / "manifest.json",
        dict(
            task=config["task"],
            classes=config["classes"],
            synthetic=True,
            split_provenance="synthetic independent fixtures",
            source="generated engineering fixture; NOT DTR training data",
            samples=rows,
        ),
    )
    return str(output / "manifest.json")


def overlap(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


def prepare_coco(source, output, config, sessions=None, trust_source_splits=False):
    """Read Roboflow COCO train/valid/test exports. Never download/upload implicitly.

    Sessions JSON maps 'train/image.jpg' to {'session': 'recording-1', 'split':'train'}.
    This must be supplied unless source splits are explicitly accepted as unverified.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if sessions is None and not trust_source_splits:
        raise ValueError("Provide --sessions or explicitly --trust-source-splits")
    session_map = read_json(sessions) if sessions else None
    aliases = {**{x: x for x in config["classes"]}, **config["aliases"]}
    rows, skipped, seen_sources = [], Counter(), {}
    rng = np.random.default_rng(config["seed"])
    for folder, default_split in [("train", "train"), ("valid", "val"), ("test", "test")]:
        directory = source / folder
        coco = read_json(directory / "_annotations.coco.json")
        if coco.get("info", {}).get("training_approved") is False:
            raise ValueError(
                "COCO source is quarantined from training; resolve its review/license gate"
            )
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        annotations = defaultdict(list)
        for ann in coco["annotations"]:
            annotations[ann["image_id"]].append(ann)
        for info in coco["images"]:
            filename = info["file_name"]
            src = (directory / filename).resolve()
            if not src.is_relative_to(directory):
                raise ValueError("COCO file path escapes source directory")
            key = f"{folder}/{filename}"
            entry = (
                session_map[key]
                if session_map is not None
                else {"session": f"unverified-{default_split}", "split": default_split}
            )
            split, session = entry["split"], entry["session"]
            if split not in SPLITS:
                raise ValueError(f"Invalid split for {key}")
            # Pixel hashes catch identical images with differing metadata/files.
            with Image.open(src) as opened:
                image = opened.convert("RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen_sources:
                if seen_sources[digest] != split:
                    raise ValueError(f"Duplicate source image across splits: {key}")
                continue
            seen_sources[digest] = split
            width, height = image.size
            boxes = []
            accepted = []
            for ann in annotations[info["id"]]:
                x, y, w, h = ann["bbox"]
                if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
                    raise ValueError(f"Malformed box in {key}")
                box = (max(0, x), max(0, y), min(width, x + w), min(height, y + h))
                boxes.append(box)  # Include unknown objects in background exclusion.
                name = categories[ann["category_id"]]
                label = aliases.get(name)
                if label is None or label == "background":
                    skipped[name] += 1
                    continue
                if min(box[2] - box[0], box[3] - box[1]) < 8:
                    continue
                pad = 0.12 * max(w, h)
                crop = (
                    max(0, int(x - pad)),
                    max(0, int(y - pad)),
                    min(width, int(x + w + pad)),
                    min(height, int(y + h + pad)),
                )
                accepted.append((crop, label))
            # Background samples come only from regions with no annotated object overlap.
            for _ in range(30):
                side = min(width, height, int(rng.integers(32, 129)))
                x = int(rng.integers(0, width - side + 1))
                y = int(rng.integers(0, height - side + 1))
                box = (x, y, x + side, y + side)
                if all(overlap(box, b) == 0 for b in boxes):
                    accepted.append((box, "background"))
                    break
            for number, (box, label) in enumerate(accepted):
                rel = f"images/{split}/{digest[:20]}-{number}.png"
                dest = output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                image.crop(box).save(dest)
                rows.append(
                    dict(
                        path=rel,
                        split=split,
                        session=session,
                        label=label,
                        sha256=sha256(dest),
                        source_image=key,
                        source_sha256=digest,
                    )
                )
    manifest = output / "manifest.json"
    write_json(
        manifest,
        dict(
            task=config["task"],
            classes=config["classes"],
            synthetic=False,
            source=str(source),
            split_provenance="session map"
            if sessions
            else "upstream splits UNVERIFIED for session leakage",
            skipped_categories=dict(skipped),
            samples=rows,
        ),
    )
    validate(manifest, config)
    return str(manifest)
