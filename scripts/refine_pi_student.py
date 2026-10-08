"""Bounded paired student refinement with visually reviewed TRAIN-only negatives.

Retains the architecture/runtime cost. Cached teacher targets are used ONLY for
the original, unchanged crops. New reviewed negatives receive hard labels only.
Control and intervention use the same starting weights, seed, optimizer and epochs.
"""
import argparse
from pathlib import Path, PurePosixPath

import numpy as np

from dtr.data import read_json, sha256, validate, write_json, load_rgb


def queue_path(queue_root):
    root = Path(queue_root)
    return root / ("queue.json" if (root / "queue.json").is_file() else "review.json")


def reviewed_negatives(queue_root, decision_path, expected_manifest):
    root = Path(queue_root).resolve()
    path = queue_path(root)
    if path.name == "queue.json":
        queue, receipt = read_json(path), read_json(root / "receipt.json")
        source_prefix = "train/"
    else:
        report = read_json(path)
        queue = report["samples"]
        receipt = dict(base_manifest_sha256=report["base_manifest_sha256"], queue_sha256=sha256(path))
        source_prefix = "balloon/train/"
    decision = read_json(decision_path)
    if (receipt["base_manifest_sha256"] != expected_manifest
            or decision["queue_sha256"] != sha256(path)
            or receipt["queue_sha256"] != decision["queue_sha256"]):
        raise ValueError("Review/manifest hash mismatch")
    indices = decision["admit"]
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(queue) for i in indices)):
        raise ValueError("Invalid review indices")
    rows = [queue[i] for i in indices]
    for row in rows:
        path = (root / row["path"]).resolve()
        if (row["split"] != "train" or row["label"] != "background"
                or not row["source_image"].startswith(source_prefix)
                or ".." in PurePosixPath(row["source_image"]).parts
                or not path.is_relative_to(root) or sha256(path) != row["sha256"]):
            raise ValueError("Invalid reviewed training crop")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--negative-queue", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-queue", help="Optional reviewed proposal-compatible training crops")
    parser.add_argument("--positive-review")
    parser.add_argument("--positive-repetitions", type=int, default=0)
    parser.add_argument("--positive-control", action="store_true",
                        help="Replace positive proposal views with same-class original crops and identical hard-label loss")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--negative-repetitions", type=int, default=8, help="0 is matched no-new-data control")
    parser.add_argument("--control-repetitions", type=int, default=0,
                        help="Repeat original crops to match intervention updates; requires negative repetitions 0")
    args = parser.parse_args()
    out, base, manifest = Path(args.output), Path(args.base_run), Path(args.manifest)
    if out.exists():
        raise FileExistsError(out)
    if (not 0 <= args.positive_repetitions <= 12
            or bool(args.positive_queue) != bool(args.positive_review)
            or bool(args.positive_queue) != bool(args.positive_repetitions)
            or (args.positive_control and not args.positive_queue)):
        parser.error("Positive queue/review require 1..12 repetitions; control requires queue")
    if not 1 <= args.epochs <= 12 or not 0 <= args.negative_repetitions <= 12:
        parser.error("Bound epochs 1..12 and negative repetitions 0..12")
    if (not 0 <= args.control_repetitions <= 12
            or (args.control_repetitions and args.negative_repetitions)):
        parser.error("Control repetitions must be 0..12 with negative repetitions 0")
    previous = read_json(base / "provenance.json")
    config = previous["config"]
    doc = validate(manifest, config, splits=("train", "val"))
    if (doc.get("synthetic", True) or doc.get("training_approved") is not True
            or previous["manifest_sha256"] != sha256(manifest)
            or previous.get("teacher_targets_kind") != "teacher_logits"):
        raise ValueError("Require approved original real crops and matching teacher cache")
    extra = reviewed_negatives(args.negative_queue, args.review, sha256(manifest))
    allowed_sources = {r.get("source_image") for r in doc["samples"] if r["split"] == "train"}
    train_sessions = {r["session"] for r in doc["samples"] if r["split"] == "train"}
    forbidden_hashes = {r["sha256"] for r in doc["samples"]
                        if r["split"] != "train" or r["label"] != "background"}
    if any(r["source_image"] not in allowed_sources or r["session"] not in train_sessions for r in extra):
        raise ValueError("Negative source is not in original training partition")
    if any(r["sha256"] in forbidden_hashes for r in extra):
        raise ValueError("Negative conflicts with an existing positive or held-out crop")
    positives = []
    if args.positive_queue:
        from scripts.build_red_blue_proposal_positives import checked_positives
        positives = checked_positives(args.positive_queue, args.positive_review, manifest)
        if {r["sha256"] for r in positives}.intersection(r["sha256"] for r in extra):
            raise ValueError("Positive/negative label conflict")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from dtr.data import batches
    from dtr.training import evaluate
    from dtr.export import export_int8
    from distill_pi_student import checked_cache, require_finite_metrics

    keras.utils.set_random_seed(config["seed"])
    rows = [r for r in doc["samples"] if r["split"] == "train"]
    k, size = len(config["classes"]), config["student_size"]
    targets = checked_cache(base, sha256(manifest), previous["teacher_sha256"], rows, config["classes"])
    targets = np.column_stack([targets, np.full(len(rows), config["alpha"], np.float32)])
    samples = [(manifest.parent / r["path"], target) for r, target in zip(rows, targets)]
    negative_target = np.zeros(2*k+1, np.float32)
    negative_target[config["classes"].index("background")] = 1
    negative_target[-1] = 1  # Ignore teacher KL for reviewed negatives, not fake logits.
    for _ in range(args.negative_repetitions):
        samples.extend((Path(args.negative_queue)/r["path"], negative_target) for r in extra)
    if args.control_repetitions:
        rng = np.random.default_rng(config["seed"])
        control = [samples[int(i)] for i in rng.choice(len(samples),
                   size=len(extra)*args.control_repetitions, replace=True)]
        samples.extend(control)
    positive_rng = np.random.default_rng(config["seed"])
    for _ in range(args.positive_repetitions):
        for row in positives:
            target = np.zeros(2*k+1, np.float32)
            target[config["classes"].index(row["label"])] = 1
            target[-1] = 1  # No cached teacher output exists for new proposal views.
            path = Path(args.positive_queue) / row["path"]
            if args.positive_control:
                original = [r for r in rows if r["label"] == row["label"]]
                path = manifest.parent / original[int(positive_rng.integers(len(original)))]["path"]
            samples.append((path, target))
    out.mkdir(parents=True)
    provenance = dict(
        config=config, student_variant=previous["student_variant"],
        base_run=str(base.resolve()), base_student_sha256=sha256(base / "student.keras"),
        manifest=str(manifest.resolve()), manifest_sha256=sha256(manifest),
        teacher_sha256=previous["teacher_sha256"], target_cache_sha256=sha256(base / "train-targets.npy"),
        negative_queue_sha256=sha256(queue_path(args.negative_queue)),
        review_sha256=sha256(args.review), script_sha256=sha256(__file__),
        negative_repetitions=args.negative_repetitions, unique_new_negatives=len(extra) if args.negative_repetitions else 0,
        control_repetitions=args.control_repetitions,
        positive_queue_sha256=sha256(Path(args.positive_queue)/"review.json") if positives else None,
        positive_review_sha256=sha256(args.positive_review) if positives else None,
        positive_repetitions=args.positive_repetitions, positive_control=args.positive_control,
        unique_positive_proposals=len(positives) if not args.positive_control else 0,
        positive_loss="hard-label cross entropy only; same-class original resampling for control",
        original_train_entries=len(rows), training_entries=len(samples), epochs=args.epochs,
        learning_rate=1e-4, augmentation="none; exact original cached targets",
        new_negative_loss="hard-label cross entropy only", synthetic_training=False,
        test_evaluated=False, deployment_approved=False, pi_zero_verified=False)
    write_json(out/"provenance.json", provenance)
    write_json(out/"status.json", dict(state="running"))
    try:
        def generate():
            for path, target in samples:
                yield load_rgb(path, size), target

        ds = tf.data.Dataset.from_generator(generate, output_signature=(
            tf.TensorSpec((size,size,3), tf.float32), tf.TensorSpec((2*k+1,), tf.float32)))
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(samples)))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.shuffle(2048, seed=config["seed"]).batch(config["batch_size"]).with_options(options).prefetch(1)
        val = batches(manifest, config, "val", size, validation_splits=("train", "val"))

        def loss(y, logits):
            hard = keras.losses.categorical_crossentropy(y[:, :k], logits, from_logits=True)
            t = config["temperature"]
            old = y[:, k:2*k]/t
            soft = tf.reduce_sum(tf.nn.softmax(old)*(tf.nn.log_softmax(old)-tf.nn.log_softmax(logits/t)),axis=-1)*t*t
            return y[:, -1]*hard + (1-y[:, -1])*soft

        def hard_loss(y, logits):
            return keras.losses.categorical_crossentropy(y[:, :k], logits, from_logits=True)

        def val_targets(x, y):
            return x, tf.concat([tf.one_hot(y,k), tf.zeros((tf.shape(y)[0],k)), tf.ones((tf.shape(y)[0],1))],axis=1)

        model = keras.models.load_model(base / "student.keras", compile=False)
        if model.input_shape != (None,size,size,3) or model.output_shape != (None,k):
            raise ValueError("Source model shape mismatch")
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=loss, metrics=[hard_loss])

        class Progress(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                require_finite_metrics(logs)

            def on_epoch_end(self, epoch, logs=None):
                require_finite_metrics(logs)
                write_json(out/"status.json", dict(state="running", epoch=epoch+1,
                                                   metrics={k:float(v) for k,v in logs.items()}))

        model.fit(ds, validation_data=val.map(val_targets), epochs=args.epochs, verbose=2,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(out/"epochs.csv")),
                             keras.callbacks.ModelCheckpoint(str(out/"best.weights.h5"),
                                 monitor="val_hard_loss", save_best_only=True, save_weights_only=True)])
        model.load_weights(out/"best.weights.h5")
        model.save(out/"student.keras")
        report = dict(validation_fp32=evaluate(model,val,config["classes"]), parameters=model.count_params(),
                      test_evaluated=False, deployment_approved=False)
        export_int8(model,manifest,config,out/"student.int8.tflite",provenance,validation_splits=("train","val"))
        if sha256(base/"student.keras") != provenance["base_student_sha256"] or sha256(manifest) != provenance["manifest_sha256"]:
            raise ValueError("Original inputs changed")
        write_json(out/"report.json", report)
        write_json(out/"status.json", dict(state="complete", deployment_approved=False))
        print(dict(accuracy=report["validation_fp32"]["accuracy"], new_negatives=provenance["unique_new_negatives"]),flush=True)
    except BaseException as exc:
        write_json(out/"status.json", dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
