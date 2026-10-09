"""Fit 129 IoU-head parameters on frozen student features; preserve class logits."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import load_rgb, read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_public_balloon_export import source_group
from scripts.build_indoor_balloon_training import admitted
from scripts.prepare_indoor_balloon_review import ARCHIVE_SHA, family


def training_rows(config):
    if config.get("training_approved") is not True or config.get("deployment_approved") is not False:
        raise ValueError("Require training-only quality config")
    manifest = config["manifest"]
    base = read_json(manifest)
    forbidden = {r["source_image"] for r in base["samples"] if r["split"] != "train"}
    rows, receipts, seen = [], [], set()
    for source in config["sources"]:
        root = Path(source["queue"]).resolve()
        queue, review = read_json(root / "review.json"), read_json(source["review"])
        if (queue["base_manifest_sha256"] != sha256(manifest) or queue["archive_sha256"] != ARCHIVE_SHA
                or queue["parent_queue_sha256"] != sha256(Path(source["parent_queue"]) / "review.json")
                or queue["parent_review_sha256"] != sha256(source["parent_review"])
                or review["queue_sha256"] != sha256(root / "review.json")
                or review.get("training_approved") is not True or review.get("deployment_approved") is not False):
            raise ValueError("Quality queue or review identity changed")
        ids = review["admit"] + review["exclude"]
        if (any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)
                or set(ids) != set(range(len(queue["samples"]))) or not review["admit"]):
            raise ValueError("Require complete explicit quality review")
        anchors = {r["id"]: r for r in admitted(source["parent_queue"], source["parent_review"], manifest)}
        for index in review["admit"]:
            row = queue["samples"][index]
            anchor = anchors.get(row["anchor_id"])
            frame = queue["frames"][row["frame_id"]]
            path, frame_path = (root / row["path"]).resolve(), (root / frame["path"]).resolve()
            f = family(Path(row["source_image"]).name)
            boxes = [row["box"], row["crop_box"], *row["target_boxes"]]
            valid = all(len(b) == 4 and np.isfinite(b).all() and
                        0 <= b[0] < b[2] <= 320 and 0 <= b[1] < b[3] <= 240 for b in boxes)
            if (not anchor or row["id"] != index or row["split"] != "train" or f is None
                    or row["source_image"] in forbidden or not valid
                    or row["source_image"] != "train/" + frame["source"]
                    or row["session"] != "engdes2:" + f
                    or row["source_group"] != source_group(frame["source"])
                    or any(row[k] != anchor[k] for k in ("source_image", "source_sha256", "frame_id", "session"))
                    or row["target_boxes"] != [t["box"] for t in frame["quality_targets"]]
                    or not path.is_relative_to(root) or not frame_path.is_relative_to(root)
                    or sha256(path) != row["sha256"] or sha256(frame_path) != frame["sha256"]
                    or index not in frame["crop_ids"]):
                raise ValueError("Invalid quality source, geometry, review or pixels")
            if row["kind"] == "exact_reviewed_background":
                expected = 0.
                if anchor["label"] != "background" or row["crop_box"] != anchor["crop_box"]:
                    raise ValueError("Unverified negative quality target")
            elif row["kind"] == "reviewed_balloon_overlap":
                expected = max((iou(row["box"], b) for b in row["target_boxes"]), default=0.)
                if anchor["label"] == "background" or expected < .02:
                    raise ValueError("Missing positive quality anchor")
            else:
                raise ValueError("Unknown quality supervision")
            if not np.isfinite(row["quality"]) or abs(row["quality"]-expected) > 1e-8:
                raise ValueError("Quality target does not match reviewed geometry")
            key = (row["source_image"], tuple(row["box"]))
            if key in seen:
                raise ValueError("Repeated quality proposal")
            seen.add(key)
            rows.append(dict(row, absolute_path=str(path)))
        receipts.append(dict(queue_sha256=sha256(root / "review.json"), review_sha256=sha256(source["review"])))
    if not rows:
        raise ValueError("Empty quality training data")
    return rows, receipts


def fit_ridge(features, targets, penalty):
    x, y = np.asarray(features, np.float64), np.asarray(targets, np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or len(x) < 2 or x.shape[1] < 1
            or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1))
            or not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("Invalid ridge inputs or penalty")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    z, offset = (x-mean)/scale, y.mean()
    w = np.linalg.solve(z.T@z + penalty*np.eye(x.shape[1]), z.T@(y-offset)) / scale
    bias = offset - mean@w
    return w.astype(np.float32), np.float32(bias)


def fold_ids(rows):
    return np.array([int(hashlib.sha256(r["source_group"].encode()).hexdigest()[:8], 16) % 5 for r in rows])


def train(config_path, student_run, output):
    output, run = Path(output), Path(student_run)
    if output.exists():
        raise FileExistsError(output)
    config = read_json(config_path)
    rows, receipts = training_rows(config)
    meta = read_json(run / "student.fp32.json")
    checkpoint = run / "student.keras"
    if (meta.get("source_student_sha256") != sha256(checkpoint)
            or meta["sha256"] != sha256(run / "student.fp32.tflite")
            or meta["classes"] != ["background", "red_balloon", "blue_balloon"]
            or meta["size"] != 64 or meta["task"] != "balloon" or meta["synthetic_training"]):
        raise ValueError("Frozen student identity mismatch")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    pupil = keras.models.load_model(checkpoint, compile=False)
    pupil.trainable = False
    features_model = keras.Model(pupil.input, pupil.get_layer("spatial_features").output)
    x = np.stack([load_rgb(r["absolute_path"], 64) for r in rows])
    features = features_model(x, training=False).numpy()
    y = np.array([r["quality"] for r in rows])
    penalty = config["ridge_penalty"]
    folds = fold_ids(rows)
    oof = np.zeros(len(rows))
    for fold in np.unique(folds):
        take = folds != fold
        w, bias = fit_ridge(features[take], y[take], penalty)
        oof[~take] = np.clip(features[~take]@w + bias, 0, 1)
    w, bias = fit_ridge(features, y, penalty)
    head = keras.layers.Dense(1, name="box_quality_raw")
    q = head(pupil.get_layer("spatial_features").output)
    combined = keras.Model(pupil.input, keras.layers.Concatenate(name="class_logits_and_quality")([pupil.output, q]))
    head.set_weights([w[:, None], np.array([bias], np.float32)])
    keras_output = combined(x, training=False).numpy()
    if not np.array_equal(keras_output[:, :3], pupil(x, training=False).numpy()):
        raise ValueError("Quality head changed frozen Keras classifier")
    output.mkdir(parents=True)
    combined.save(output / "quality.keras")

    @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
    def serve(inputs):
        return combined(inputs, training=False)

    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
    path = output / "quality.fp32.tflite"
    path.write_bytes(binary)
    interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    values = []
    for crop in x:
        interpreter.set_tensor(inp["index"], crop[None])
        interpreter.invoke()
        values.append(interpreter.get_tensor(out["index"])[0])
    values = np.asarray(values)
    max_delta = float(np.abs(values-keras_output).max())
    if max_delta > .002:
        raise ValueError("Combined TFLite export parity failed")
    prediction = np.clip(values[:, 3], 0, 1)
    metadata = dict(kind="balloon_quality_research", classes=meta["classes"], size=64,
        threshold=.8, output="first 3: class logits; fourth: raw IoU regression, clip to [0,1]",
        tensor_dtype="float32", sha256=sha256(path), bytes=len(binary),
        source_student_sha256=sha256(checkpoint), source_tflite_sha256=meta["sha256"],
        training_config_sha256=sha256(config_path), source_reviews=receipts,
        script_sha256=sha256(__file__), parameters=combined.count_params(),
        added_parameters=head.count_params(), training_samples=len(rows), ridge_penalty=penalty,
        test_evaluated=False, synthetic_training=False, deployment_approved=False, pi_zero_verified=False)
    write_json(path.with_suffix(".json"), metadata)
    report = dict(train_mae=float(np.abs(prediction-y).mean()),
        grouped_head_only_oof_mae=float(np.abs(oof-y).mean()),
        oof_caveat="Only quality head held out; backbone already trained on related crops/environments",
        max_keras_tflite_delta=max_delta, frozen_keras_logits_identical=True,
        parameters=combined.count_params(), added_parameters=head.count_params(), bytes=len(binary),
        examples=[dict(source_group=r["source_group"], target=float(t), fit=float(p), oof=float(o), fold=int(f))
                  for r, t, p, o, f in zip(rows, y, prediction, oof, folds)],
        test_evaluated=False, deployment_approved=False, pi_timing_measured=False)
    write_json(output / "report.json", report)
    print({k: v for k, v in report.items() if k != "examples"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "student-run", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    train(args.config, args.student_run, args.output)
