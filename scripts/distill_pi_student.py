"""Distill an existing real V4 teacher to an integration-only INT8 crop student.

Cache teacher logits once; do not retrain teachers or open reserved-test images.
No camera, SSH, UART or flight commands. No accuracy gate blocks research export.
"""

import argparse
from pathlib import Path
import time

import numpy as np

from dtr.data import load_rgb, read_json, sha256, validate, write_json


def require_finite_metrics(logs):
    if not logs or not all(np.isfinite(float(value)) for value in logs.values()):
        raise ValueError("Missing or non-finite training metrics; refusing export")


def checked_inputs(manifest, config, teacher_path):
    doc = validate(manifest, config, splits=("train", "val"))
    meta = read_json(Path(teacher_path).with_suffix(".json"))
    if doc.get("synthetic", True) or doc.get("training_approved") is False:
        raise ValueError("Require reviewed real training data")
    if (meta.get("kind") != "keras_teacher" or meta.get("synthetic_training", True)
            or meta.get("sha256") != sha256(teacher_path)
            or meta.get("task") != config["task"] or meta.get("classes") != config["classes"]
            or meta.get("size") != config["teacher_size"]):
        raise ValueError("Teacher identity, taxonomy, size or checksum mismatch")
    return doc, meta


def checked_cache(run, manifest_hash, teacher_hash, rows, classes):
    """Reuse only train logits with matching source identities, order and labels."""
    run = Path(run)
    provenance = read_json(run / "provenance.json")
    kind = provenance.get("teacher_targets_kind")
    if kind != "teacher_logits" and (kind is not None or provenance.get("alpha", 1) >= 1):
        raise ValueError("Cache does not contain verified teacher logits")
    if (provenance.get("manifest_sha256") != manifest_hash
            or provenance.get("teacher_sha256") != teacher_hash
            or provenance.get("config", {}).get("classes") != classes):
        raise ValueError("Teacher cache source identity mismatch")
    targets = np.load(run / "train-targets.npy", allow_pickle=False)
    k = len(classes)
    hard = np.eye(k, dtype=np.float32)[[classes.index(row["label"]) for row in rows]]
    if (targets.shape != (len(rows), 2*k) or not np.isfinite(targets).all()
            or not np.array_equal(targets[:, :k], hard)):
        raise ValueError("Invalid teacher cache targets or row labels")
    return targets.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--student-variant",
                        choices=("tiny", "spatial", "context", "separable_context"), default="tiny")
    parser.add_argument("--alpha", type=float, help="Hard-label weight; 1 disables distillation")
    parser.add_argument("--size", type=int, help="Crop input size; does not add source detail")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-cache-run", help="Reuse hash-bound train-only teacher logits")
    parser.add_argument("--initialize-context", help="Existing context run for SVD warm-start; separable_context only")
    parser.add_argument("--initialize-student", help="Existing same-architecture student; approved new training manifest allowed")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not 1e-6 <= args.learning_rate <= 1e-2 or (args.initialize_context and args.initialize_student):
        parser.error("Require learning rate 1e-6..1e-2 and at most one initialization source")
    if not 1 <= args.epochs <= 25:
        parser.error("Choose 1..25 epochs")
    config = read_json(args.config)
    for key, value in (("alpha", args.alpha), ("student_size", args.size), ("seed", args.seed)):
        if value is not None:
            config[key] = value
    if not 0 <= config["alpha"] <= 1 or config["temperature"] <= 0:
        parser.error("Require alpha in [0,1] and positive temperature")
    if config["student_size"] < 32 or config["student_size"] % 16:
        parser.error("Require crop size >=32 divisible by 16")
    doc, meta = checked_inputs(args.manifest, config, args.teacher)
    initial_path, initial_hash = None, None
    if args.initialize_context:
        if args.student_variant != "separable_context":
            parser.error("Context warm-start requires separable_context")
        initial_run = Path(args.initialize_context)
        previous = read_json(initial_run / "provenance.json")
        if (previous.get("student_variant") != "context"
                or previous.get("teacher_sha256") != meta["sha256"]
                or previous.get("manifest_sha256") != sha256(args.manifest)
                or previous["config"]["classes"] != config["classes"]):
            raise ValueError("Warm-start data/teacher/taxonomy mismatch")
        initial_path = initial_run / "student.keras"
        initial_hash = sha256(initial_path)
    if args.initialize_student:
        initial_run = Path(args.initialize_student)
        previous = read_json(initial_run / "provenance.json")
        if (previous.get("student_variant") != args.student_variant
                or previous["config"]["classes"] != config["classes"]
                or previous["config"]["student_size"] != config["student_size"]
                or previous["config"]["task"] != config["task"]):
            raise ValueError("Student warm-start architecture/taxonomy/input mismatch")
        initial_path = initial_run / "student.keras"
        initial_hash = sha256(initial_path)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from dtr.models import research_student
    from dtr.training import evaluate
    from dtr.export import export_int8
    from dtr.data import batches
    from dtr.runtime import Predictor

    keras.utils.set_random_seed(config["seed"])
    output.mkdir(parents=True)
    provenance = dict(teacher=str(Path(args.teacher).resolve()), teacher_sha256=meta["sha256"],
                      manifest=str(Path(args.manifest).resolve()), manifest_sha256=sha256(args.manifest),
                      config=config, epochs=args.epochs, temperature=config["temperature"],
                      alpha=config["alpha"], augmentation="none; cached logits for identical source crops",
                      student_variant=args.student_variant,
                      training_script_sha256=sha256(__file__),
                      models_source_sha256=sha256(Path(__file__).resolve().parents[1] / "src/dtr/models.py"),
                      target_cache_run=args.target_cache_run,
                      initialize_context=args.initialize_context, initial_student_sha256=initial_hash,
                      initialize_student=args.initialize_student, learning_rate=args.learning_rate,
                      teacher_targets_kind=("teacher_logits" if args.target_cache_run
                                            or config["alpha"] < 1 else "unused_zeros"),
                      test_evaluated=False, deployment_approved=False, pi_zero_verified=False,
                      purpose="integration_research", tensorflow=tf.__version__, keras=keras.__version__)
    write_json(output / "provenance.json", provenance)
    write_json(output / "status.json", dict(state="running", stage="teacher_logit_cache"))
    try:
        k = len(config["classes"])
        root = Path(args.manifest).resolve().parent
        rows = [r for r in doc["samples"] if r["split"] == "train"]
        targets = np.empty((len(rows), 2*k), dtype=np.float32)
        batch_size = config["batch_size"]
        if args.target_cache_run:
            targets = checked_cache(args.target_cache_run, provenance["manifest_sha256"],
                                    meta["sha256"], rows, config["classes"])
            provenance["target_cache_sha256"] = sha256(Path(args.target_cache_run) / "train-targets.npy")
            write_json(output / "provenance.json", provenance)
        elif config["alpha"] == 1:
            targets.fill(0)
            targets[:, :k] = np.eye(k)[[config["classes"].index(r["label"]) for r in rows]]
        else:
            mentor = keras.models.load_model(args.teacher, compile=False)
            if mentor.input_shape != (None, config["teacher_size"], config["teacher_size"], 3) or mentor.output_shape != (None, k):
                raise ValueError("Loaded teacher shape mismatch")
            for start in range(0, len(rows), batch_size):
                group = rows[start:start+batch_size]
                x = np.stack([load_rgb(root / r["path"], config["teacher_size"]) for r in group])
                logits = mentor(x, training=False).numpy()
                if not np.isfinite(logits).all():
                    raise ValueError("Non-finite teacher targets")
                hard = np.eye(k, dtype=np.float32)[[config["classes"].index(r["label"]) for r in group]]
                targets[start:start+len(group)] = np.concatenate([hard, logits], axis=1)
                if start % (batch_size*32) == 0:
                    print("Teacher targets", start, "of", len(rows), flush=True)
            del mentor
        np.save(output / "train-targets.npy", targets)
        keras.backend.clear_session()
        # Cache use and teacher loading must not change student initialization/shuffle.
        keras.utils.set_random_seed(config["seed"])

        def generate():
            for row, target in zip(rows, targets):
                yield load_rgb(root / row["path"], config["student_size"]), target

        size = config["student_size"]
        ds = tf.data.Dataset.from_generator(generate, output_signature=(
            tf.TensorSpec((size, size, 3), tf.float32), tf.TensorSpec((2*k,), tf.float32)))
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(rows)))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.shuffle(min(len(rows), 2048), seed=config["seed"]).batch(batch_size).with_options(options).prefetch(1)
        val = batches(args.manifest, config, "val", size, validation_splits=("train", "val"))
        # Selection uses val_hard_loss, never the padded-logit validation KL loss.
        def padded_labels(x, y):
            return x, tf.concat([tf.one_hot(y, k), tf.zeros((tf.shape(y)[0], k))], axis=1)

        temperature, alpha = config["temperature"], config["alpha"]
        if temperature <= 0 or not 0 <= alpha <= 1:
            raise ValueError("Invalid distillation configuration")

        def loss(y, logits):
            hard = keras.losses.categorical_crossentropy(y[:, :k], logits, from_logits=True)
            teacher_logits = y[:, k:]
            p = tf.nn.softmax(teacher_logits / temperature)
            soft = tf.reduce_sum(p * (tf.nn.log_softmax(teacher_logits / temperature)
                                     - tf.nn.log_softmax(logits / temperature)), axis=-1) * temperature**2
            return alpha * hard + (1-alpha) * soft

        def hard_loss(y, logits):
            return keras.losses.categorical_crossentropy(y[:, :k], logits, from_logits=True)

        pupil = research_student(k, size, args.student_variant)
        if args.initialize_student:
            initial = keras.models.load_model(initial_path, compile=False)
            pupil.set_weights(initial.get_weights())
            provenance["initialization"] = dict(kind="same-architecture checkpoint",
                parent_manifest_sha256=previous["manifest_sha256"], teacher_logits_reused=bool(args.target_cache_run))
            write_json(output / "provenance.json", provenance)
            write_json(output / "initial-validation.json", evaluate(pupil,val,config["classes"]))
            del initial
        elif initial_path is not None:
            from dtr.models import initialize_separable_context
            initial = keras.models.load_model(initial_path, compile=False)
            provenance["initialization"] = initialize_separable_context(initial, pupil)
            write_json(output / "provenance.json", provenance)
            write_json(output / "initial-validation.json", evaluate(pupil, val, config["classes"]))
            del initial
        pupil.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss=loss, metrics=[hard_loss])

        class Progress(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                require_finite_metrics(logs)

            def on_epoch_end(self, epoch, logs=None):
                require_finite_metrics(logs)
                pupil.save(output / "latest.keras")
                write_json(output / "status.json", dict(state="running", stage="distillation",
                           epoch=epoch+1, epochs=args.epochs, metrics={a: float(b) for a,b in logs.items()}))

        pupil.fit(ds, validation_data=val.map(padded_labels), epochs=args.epochs, verbose=2,
                  callbacks=[Progress(), keras.callbacks.ModelCheckpoint(str(output / "best.weights.h5"),
                             monitor="val_hard_loss", save_best_only=True, save_weights_only=True),
                             keras.callbacks.CSVLogger(str(output / "epochs.csv"))])
        pupil.load_weights(output / "best.weights.h5")
        pupil.save(output / "student.keras")
        fp32 = evaluate(pupil, val, config["classes"])
        write_json(output / "status.json", dict(state="running", stage="int8_export"))
        metadata = export_int8(pupil, args.manifest, config, output / "student.int8.tflite",
                               provenance, validation_splits=("train", "val"))
        predictor = Predictor(output / "student.int8.tflite", allow_unvalidated=True)
        confusion = np.zeros((k, k), dtype=int)
        for row in doc["samples"]:
            if row["split"] != "val":
                continue
            prediction = predictor.predict(root / row["path"])
            confusion[config["classes"].index(row["label"]), config["classes"].index(prediction["label"])] += 1
        if sha256(args.teacher) != meta["sha256"] or sha256(args.manifest) != provenance["manifest_sha256"]:
            raise ValueError("Inputs changed during distillation")
        if initial_path is not None and sha256(initial_path) != initial_hash:
            raise ValueError("Warm-start checkpoint changed during training")
        report = dict(status="INTEGRATION_CANDIDATE_UNVALIDATED", student_parameters=pupil.count_params(),
                      validation_fp32=fp32, validation_int8=dict(
                          accuracy=float(confusion.trace()/confusion.sum()),
                          confusion_true_rows_predicted_columns=confusion.tolist()),
                      export=metadata, test_evaluated=False, deployment_approved=False,
                      pi_zero_verified=False, scope="Crop validation only; not full-frame detector accuracy")
        write_json(output / "report.json", report)
        write_json(output / "status.json", dict(state="complete", result=report["status"], completed_at=time.time()))
        print({"status": report["status"], "parameters": pupil.count_params(), "bytes": metadata["bytes"],
               "validation_int8_accuracy": report["validation_int8"]["accuracy"]}, flush=True)
    except BaseException as exc:
        write_json(output / "status.json", dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
