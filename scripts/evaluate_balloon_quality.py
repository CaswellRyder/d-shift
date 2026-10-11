"""Research-only four-output quality model and hash-bound scene comparison."""
import argparse
import hashlib
import io
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.compare_balloon_containment import area, before_parts, confidence_ordered, nested
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.research_balloon_search import suppress_confirmed_balloon_parts


class QualityPredictor:
    def __init__(self, path):
        self.metadata = read_json(Path(path).with_suffix(".json"))
        m = self.metadata
        if (m.get("kind") != "balloon_quality_research" or m["sha256"] != sha256(path)
                or m["classes"] != ["background", "red_balloon", "blue_balloon"]
                or m["size"] != 64 or m["threshold"] != .8 or m.get("deployment_approved") is not False):
            raise ValueError("Invalid research quality model identity")
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input, self.output = self.interpreter.get_input_details()[0], self.interpreter.get_output_details()[0]
        if (self.input["shape"].tolist() != [1, 64, 64, 3] or self.output["shape"].tolist() != [1, 4]
                or self.input["dtype"] != np.float32 or self.output["dtype"] != np.float32):
            raise ValueError("Quality model requires explicit four-output float contract")

    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Require nonempty uint8 RGB crop")
        resized = Image.fromarray(rgb).resize((64, 64), Image.Resampling.BILINEAR)
        values = np.asarray(resized, np.float32)[None]
        self.interpreter.set_tensor(self.input["index"], values)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output["index"])[0]
        if not np.isfinite(result).all():
            raise ValueError("Non-finite quality model output")
        return softmax(result[:3]), float(np.clip(result[3], 0, 1))


def quality_ordered(observations):
    rows = before_parts(observations)
    if any(not np.isfinite(r["box_quality"]) or not 0 <= r["box_quality"] <= 1 for r in rows):
        raise ValueError("Invalid box-quality score")
    indices = [i for i, r in enumerate(rows) if r["accepted"] and r["label"] in LABELS]
    indices.sort(key=lambda i: (-rows[i]["box_quality"], -rows[i]["score"], -area(rows[i]["box"]), i))
    kept = []
    for index in indices:
        row = rows[index]
        winner = next((j for j in kept if rows[j]["label"] == row["label"]
                       and nested(rows[j]["box"], row["box"])), None)
        if winner is None:
            kept.append(index)
        else:
            row.update(accepted=False, suppressed=True, suppression_reason="quality_ordered_containment",
                       suppressed_by_index=winner)
    return rows


def quality_gated_ordered(observations, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid fixed quality threshold")
    rows = before_parts(observations)
    for row in rows:
        if not np.isfinite(row["box_quality"]) or not 0 <= row["box_quality"] <= 1:
            raise ValueError("Invalid box-quality score")
        if row["accepted"] and row["label"] in LABELS and row["box_quality"] < threshold:
            row.update(accepted=False, suppressed=True, suppression_reason="training_calibrated_quality_gate")
    return quality_ordered(rows)


def panel_frames(panel, report, queue=None, review_path=None):
    if panel == "reviewed":
        if not queue or not review_path:
            raise ValueError("Reviewed panels require explicit queue and review")
        if (report["queue_sha256"] != sha256(Path(queue) / "review.json")
                or report["review_sha256"] != sha256(review_path)):
            raise ValueError("Reviewed development panel changed")
        frames, _ = reviewed_frames(queue, review_path)
        return {row["source"]: (rgb, row["provisional_truth"]) for row, rgb in frames}
    if queue is not None or review_path is not None:
        raise ValueError("Explicit queue/review requires reviewed panel mode")
    if panel == "indoor":
        root = Path("data/engdes2-development-scenes-20261008")
        review = "configs/engdes2-development-scenes-20261008.json"
        if report["queue_sha256"] != sha256(root / "review.json") or report["review_sha256"] != sha256(review):
            raise ValueError("Indoor panel changed")
        frames, _ = reviewed_frames(root, review)
        return {row["source"]: (rgb, row["provisional_truth"]) for row, rgb in frames}
    if panel != "original":
        raise ValueError("Unknown development panel")
    manifest = Path("data/balloon-red-blue-bootstrap-20261008/manifest.json")
    review_path = "configs/balloon-red-blue-development-scenes.json"
    if report["manifest_sha256"] != sha256(manifest) or report["scene_review_sha256"] != sha256(review_path):
        raise ValueError("Original panel changed")
    doc, review = read_json(manifest), read_json(review_path)
    archive_path = "data/raw/matterport-balloon/balloon_dataset.zip"
    if sha256(archive_path) != doc["source_archive_sha256"]:
        raise ValueError("Original archive changed")
    val = [r for r in doc["samples"] if r["split"] == "val"]
    if set(r["source_image"] for r in val) != set(review["sources"]):
        raise ValueError("Development source identities changed")
    frames = {}
    with zipfile.ZipFile(archive_path) as archive:
        for source in review["sources"]:
            rows = [r for r in val if r["source_image"] == source]
            with Image.open(io.BytesIO(archive.read(source))) as im:
                im = im.convert("RGB")
                if any(r["source_sha256"] != hashlib.sha256(im.tobytes()).hexdigest() for r in rows):
                    raise ValueError("Original development pixels changed")
                rgb = cv2.resize(np.asarray(im), (320, 240), interpolation=cv2.INTER_AREA)
                truth = [dict(label=r["label"], box=[v*(320/im.width if i % 2 == 0 else 240/im.height)
                         for i, v in enumerate(r["box_xyxy"])]) for r in rows if r["label"] in LABELS]
                frames[source] = (rgb, truth)
    return frames


def evaluate(model_path, source_path, name, panel, queue=None, review_path=None,
             variant="mser_confirmed_parts", gate_config=None):
    source = read_json(source_path)
    if source.get("test_evaluated") is not False or source.get("deployment_approved") is not False:
        raise ValueError("Require development-only source report")
    if variant not in ("mser_confirmed_parts", "mser_blue_cores"):
        raise ValueError("Unsupported frozen proposal variant")
    if panel == "reviewed":
        if not queue or not review_path:
            raise ValueError("Reviewed panels require explicit queue and review")
        # Validate every saved frame, orientation, truth and count before replay.
        from scripts.audit_balloon_pipeline_errors import audit
        audit([source_path], queue, review_path)
    predictor = QualityPredictor(model_path)
    threshold = None
    if gate_config:
        from scripts.calibrate_balloon_quality_gate import load_gate
        threshold = load_gate(gate_config, model_path)
    baseline = source["results"][name]
    if baseline["model_sha256"] != predictor.metadata["source_tflite_sha256"] or baseline["threshold"] != .8:
        raise ValueError("Classifier does not match original scene report")
    pixels = panel_frames(panel, source, queue, review_path)
    old = baseline["full_frame"][variant]
    if {f["source"] for f in old["frames"]} != set(pixels):
        raise ValueError("Scene report coverage mismatch")
    enriched, max_delta = [], 0.
    for frame in old["frames"]:
        rgb, truth = pixels[frame["source"]]
        if frame["truth"] != truth:
            raise ValueError("Reviewed truth changed")
        rows = []
        for row in frame["detections"]:
            a, b, c, d = row["crop_box"]
            if not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
                raise ValueError("Invalid saved crop box")
            scores, q = predictor.predict(rgb[b:d, a:c])
            delta = float(np.abs(scores-np.asarray(row["scores"])).max())
            max_delta = max(delta, max_delta)
            best = int(scores.argmax())
            if (delta > 1e-5 or predictor.metadata["classes"][best] != row["label"]
                    or bool(best != 0 and scores[best] >= .8) != row["raw_accepted"]):
                raise ValueError("Class predictions changed during quality export")
            rows.append(dict(row, box_quality=q))
        enriched.append(dict(frame, detections=rows))
    results = {}
    policies = [("parent_first", lambda rows: suppress_confirmed_balloon_parts(before_parts(rows))),
                ("confidence_ordered", confidence_ordered), ("quality_ordered", quality_ordered)]
    if threshold is not None:
        policies.append(("quality_gated_ordered", lambda rows: quality_gated_ordered(rows, threshold)))
    for kind, select in policies:
        total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
        frames = []
        for frame in enriched:
            detections = select(frame["detections"])
            counts = detection_counts(frame["truth"], detections)
            for label in LABELS:
                for key in ("tp", "fp", "fn"):
                    total[label][key] += counts[label][key]
            frames.append(dict(frame, detections=detections, counts=counts))
        results[kind] = dict(metrics=metrics(total), frames=frames)
    if results["parent_first"]["metrics"] != old["metrics"]:
        raise ValueError("Original full-scene metrics did not reproduce")
    return dict(results=results, model_sha256=sha256(model_path), metadata_sha256=sha256(Path(model_path).with_suffix(".json")),
        source_report_sha256=sha256(source_path), source_model=name, panel=panel,
        source_variant=variant, source_selection=source.get("selection"),
        source_split_max_aspect=source.get("split_max_aspect"),
        queue_sha256=source.get("queue_sha256"), review_sha256=source.get("review_sha256"),
        quality_gate_sha256=sha256(gate_config) if gate_config else None, quality_gate_threshold=threshold,
        max_class_probability_delta=max_delta, script_sha256=sha256(__file__),
        scope="Development quality-head ranking; old proposals and class acceptance preserved",
        test_evaluated=False, deployment_approved=False, pi_timing_measured=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source-report", "source-model", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--panel", choices=("indoor", "original", "reviewed"), required=True)
    parser.add_argument("--queue")
    parser.add_argument("--review")
    parser.add_argument("--variant", choices=("mser_confirmed_parts", "mser_blue_cores"), default="mser_confirmed_parts")
    parser.add_argument("--quality-gate")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = evaluate(args.model, args.source_report, args.source_model, args.panel,
                      args.queue, args.review, args.variant, args.quality_gate)
    write_json(args.output, result)
    print({k: v["metrics"] for k, v in result["results"].items()})
