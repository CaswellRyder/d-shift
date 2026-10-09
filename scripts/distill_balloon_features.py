"""Paired logit/feature distillation; training-only projection is never exported."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import load_rgb, read_json, sha256, write_json
from scripts.distill_pi_student import checked_inputs

MANIFEST_SHA = "6011ae406f036759d9e6e4b334b5dd73de82b61248160d471c409ffede7e4cbb"
TEACHER_SHA = "db57446ecfba29c29353924792908d9a762f5452f50664f66c43cfdcc0ff45fe"


def training_boundary(doc):
    if doc.get("training_approved") is not True or doc.get("synthetic") is not False:
        raise ValueError("Require explicitly approved real training data")
    rows = [r for r in doc["samples"] if r["split"] == "train"]
    for row in rows:
        source = row.get("source_image")
        if not source or Path(source).name.lower().startswith("img_"):
            raise ValueError("Missing source or reserved IMG family in training")
    if not rows:
        raise ValueError("No training rows")
    return rows


def normalize_features(values):
    values = np.asarray(values, np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Require finite nonempty feature matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def weights_hash(model):
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(str(weight.shape).encode())
        digest.update(np.ascontiguousarray(weight).tobytes())
    return digest.hexdigest()


def checked_targets(path, manifest_hash, teacher_hash, count):
    path = Path(path)
    receipt = read_json(path / "receipt.json")
    if (
        receipt["manifest_sha256"] != manifest_hash
        or receipt["teacher_sha256"] != teacher_hash
        or receipt["split"] != "train"
        or receipt["feature_layer"] != "flatten"
        or receipt["targets_sha256"] != sha256(path / "targets.npz")
    ):
        raise ValueError("Feature cache identity mismatch")
    with np.load(path / "targets.npz", allow_pickle=False) as data:
        logits, features = data["logits"], data["features"]
    if (
        logits.shape != (count, 3)
        or features.shape != (count, 1280)
        or not np.isfinite(logits).all()
        or not np.isfinite(features).all()
        or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid feature cache arrays")
    return logits.astype(np.float32), features.astype(np.float32), receipt


def prepare_cache(path, teacher_path, manifest, root, rows, keras):
    path = Path(path)
    if path.exists():
        return checked_targets(path, sha256(manifest), sha256(teacher_path), len(rows))
    mentor = keras.models.load_model(teacher_path, compile=False)
    if mentor.input_shape != (None, 96, 96, 3) or mentor.output_shape != (None, 3):
        raise ValueError("Teacher shape changed")
    feature = mentor.get_layer("flatten").output
    if feature.shape[-1] != 1280:
        raise ValueError("Teacher feature width changed")
    extractor = keras.Model(mentor.input, [mentor.output, feature])
    logits, features = [], []
    for start in range(0, len(rows), 16):
        x = np.stack([load_rgb(root / r["path"], 96) for r in rows[start : start + 16]])
        score, hint = extractor(x, training=False)
        logits.append(score.numpy())
        features.append(hint.numpy())
    logits, features = np.concatenate(logits), normalize_features(np.concatenate(features))
    path.mkdir(parents=True, exist_ok=False)
    np.savez(path / "targets.npz", logits=logits, features=features)
    write_json(
        path / "receipt.json",
        dict(
            manifest_sha256=sha256(manifest),
            teacher_sha256=sha256(teacher_path),
            targets_sha256=sha256(path / "targets.npz"),
            split="train",
            feature_layer="flatten",
            rows=len(rows),
            row_paths=[r["path"] for r in rows],
            preprocessing="Teacher 96x96 RGB Pillow bilinear; unit-L2 per-example feature",
            script_sha256=sha256(__file__),
            test_evaluated=False,
        ),
    )
    return checked_targets(path, sha256(manifest), sha256(teacher_path), len(rows))


def train(args):
    if (
        args.seed not in (42, 43)
        or args.feature_weight not in (0.0, 1.0)
        or not 1 <= args.epochs <= 25
    ):
        raise ValueError("Fixed paired experiment: seed 42/43, feature weight 0/1, epochs 1..25")
    output, root = Path(args.output), Path(args.manifest).resolve().parent
    if output.exists():
        raise FileExistsError(output)
    if sha256(args.manifest) != MANIFEST_SHA or sha256(args.teacher) != TEACHER_SHA:
        raise ValueError("This experiment requires the frozen reviewed manifest and teacher")
    rows = training_boundary(read_json(args.manifest))  # Before opening any pixels.
    config = read_json("configs/balloon-red-blue.json")
    if (
        config["classes"] != ["background", "red_balloon", "blue_balloon"]
        or config["student_size"] != 64
        or config["teacher_size"] != 96
        or config["threshold"] != 0.8
        or config["temperature"] != 4
        or config["alpha"] != 0.5
    ):
        raise ValueError("Frozen training contract changed")
    doc, _ = checked_inputs(args.manifest, config, args.teacher)
    config = dict(config, seed=args.seed, batch_size=16)
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from dtr.models import research_student
    from dtr.export import export_int8

    logits, hints, cache = prepare_cache(args.cache, args.teacher, args.manifest, root, rows, keras)
    x = np.stack([load_rgb(root / r["path"], 64) for r in rows])
    labels = np.eye(3, dtype=np.float32)[[config["classes"].index(r["label"]) for r in rows]]
    val_rows = [r for r in doc["samples"] if r["split"] == "val"]
    vx = np.stack([load_rgb(root / r["path"], 64) for r in val_rows])
    vy = np.eye(3, dtype=np.float32)[[config["classes"].index(r["label"]) for r in val_rows]]
    keras.backend.clear_session()
    keras.utils.set_random_seed(args.seed)
    pupil = research_student(3, 64, "separable_context")
    initial_hash = weights_hash(pupil)
    # Present in BOTH arms for matched initialization; never part of pupil output.
    projection = keras.layers.Dense(1280, name="training_only_hint_projection")(
        pupil.get_layer("spatial_features").output
    )
    trainer = keras.Model(pupil.input, [pupil.output, projection])
    optimizer = keras.optimizers.Adam(0.001)

    @tf.function
    def step(images, hard, teacher_logits, teacher_features):
        with tf.GradientTape() as tape:
            predicted, projected = trainer(images, training=True)
            ce = tf.reduce_mean(
                keras.losses.categorical_crossentropy(hard, predicted, from_logits=True)
            )
            p = tf.nn.softmax(teacher_logits / 4)
            kl = (
                tf.reduce_mean(
                    tf.reduce_sum(
                        p
                        * (
                            tf.nn.log_softmax(teacher_logits / 4) - tf.nn.log_softmax(predicted / 4)
                        ),
                        axis=1,
                    )
                )
                * 16
            )
            hint = tf.reduce_mean(
                1
                - tf.reduce_sum(tf.math.l2_normalize(projected, axis=1) * teacher_features, axis=1)
            )
            total = 0.5 * ce + 0.5 * kl + args.feature_weight * hint
        gradients = tape.gradient(total, trainer.trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainer.trainable_weights))
        return total, ce, kl, hint

    provenance = dict(
        config=config,
        manifest_sha256=MANIFEST_SHA,
        teacher_sha256=TEACHER_SHA,
        feature_weight=args.feature_weight,
        initial_weights_sha256=initial_hash,
        teacher_cache=cache,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=0.001,
        teacher_feature="flatten:1280",
        student_feature="spatial_features:128",
        hint_loss="mean cosine distance, teacher features unit-L2",
        class_loss="0.5 hard CE + 0.5 KL at temperature 4, multiplied by T squared",
        student_variant="separable_context",
        feature_projection_exported=False,
        train_examples=len(rows),
        validation_examples=len(val_rows),
        selection="minimum validation hard-label cross entropy; no development-scene selection",
        shuffle="numpy default_rng(seed).permutation each epoch; matched across arms",
        augmentation="none; teacher/student share source crop, different fixed input sizes",
        training_script_sha256=sha256(__file__),
        models_source_sha256=sha256("src/dtr/models.py"),
        input_guard_sha256=sha256("scripts/distill_pi_student.py"),
        test_evaluated=False,
        deployment_approved=False,
        pi_zero_verified=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", provenance)
    write_json(output / "status.json", dict(state="running", stage="training"))
    best, best_epoch, history = float("inf"), None, []
    rng = np.random.default_rng(args.seed)
    try:
        for epoch in range(args.epochs):
            order = rng.permutation(len(rows))
            totals = np.zeros(4)
            for start in range(0, len(rows), 16):
                ids = order[start : start + 16]
                losses = np.asarray(
                    [float(v) for v in step(x[ids], labels[ids], logits[ids], hints[ids])]
                )
                if not np.isfinite(losses).all():
                    raise ValueError("Non-finite training loss")
                totals += losses * len(ids)
            predicted = pupil(vx, training=False)
            val_loss = float(
                tf.reduce_mean(
                    keras.losses.categorical_crossentropy(vy, predicted, from_logits=True)
                )
            )
            if not np.isfinite(val_loss):
                raise ValueError("Non-finite validation loss")
            if val_loss < best:
                best, best_epoch = val_loss, epoch + 1
                pupil.save(output / "student.keras")
            record = dict(
                epoch=epoch + 1,
                training_losses=(totals / len(rows)).tolist(),
                val_hard_loss=val_loss,
            )
            history.append(record)
            write_json(output / "history.json", history)
            write_json(output / "status.json", dict(state="running", stage="training", **record))
            print(record, flush=True)
        pupil = keras.models.load_model(output / "student.keras", compile=False)
        if pupil.output_shape != (None, 3) or pupil.count_params() != 7763:
            raise ValueError("Training-only projection leaked into student export")
        if sha256(args.teacher) != TEACHER_SHA or sha256(args.manifest) != MANIFEST_SHA:
            raise ValueError("Frozen inputs changed during training")
        export_int8(
            pupil,
            args.manifest,
            config,
            output / "student.int8.tflite",
            provenance,
            validation_splits=("train", "val"),
        )
        write_json(
            output / "report.json",
            dict(
                best_epoch=best_epoch,
                best_val_hard_loss=best,
                student_parameters=pupil.count_params(),
                training_projection_parameters=128 * 1280 + 1280,
                final_feature_loss=history[-1]["training_losses"][3],
                source_student_sha256=sha256(output / "student.keras"),
                test_evaluated=False,
                deployment_approved=False,
                qualification_established=False,
            ),
        )
        write_json(output / "status.json", dict(state="complete", stage="training_and_int8_export"))
    except BaseException as exc:
        write_json(output / "status.json", dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "teacher", "cache", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--feature-weight", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    train(parser.parse_args())
