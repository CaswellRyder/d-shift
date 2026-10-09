"""Joint box-quality refinement versus matched frozen-feature head-only control.

The classifier head is frozen; joint mode updates shared features under a parent
logit-preservation objective. Quality-label fold zero is excluded from refinement
inputs, not just from the auxiliary loss. No IMG fitting or reserved-test pixels.
"""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import load_rgb, read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.build_paired_balloon_augmentation import checked_base
from scripts.train_balloon_quality import fit_ridge, fold_ids, training_rows


def split_refinement(class_rows, quality_rows, holdout_fold=0):
    if type(holdout_fold) is not int or not 0 <= holdout_fold < 5:
        raise ValueError("Invalid quality-label holdout fold")
    folds = fold_ids(quality_rows)
    qtrain = [r for r, f in zip(quality_rows, folds) if f != holdout_fold]
    qval = [r for r, f in zip(quality_rows, folds) if f == holdout_fold]
    if len(qtrain) < 2 or not qval:
        raise ValueError("Require quality training and holdout examples")
    held_sources = {r["source_image"] for r in qval}
    held_groups = {r["source_group"] for r in qval}
    held_hashes = {r["source_sha256"] for r in qval}
    train = [r for r in class_rows if r["split"] == "train"
             and r.get("source_image") not in held_sources
             and r.get("source_group") not in held_groups
             and r.get("source_sha256") not in held_hashes]
    if not train or any(r["source_image"] in held_sources or r["source_sha256"] in held_hashes for r in qtrain):
        raise ValueError("Empty class pool or cross-fold quality-source leakage")
    return train, qtrain, qval


def check_config(config, mode):
    if (mode not in ("joint", "head_only") or config["deployment_approved"] is not False
            or type(config["epochs"]) is not int or not 1 <= config["epochs"] <= 25
            or type(config["batch_size"]) is not int or not 1 <= config["batch_size"] <= 64
            or not 1e-6 <= config["learning_rate"] <= 1e-3
            or not 0 < config["temperature"] <= 10
            or config["quality_weight"] != 1 or config["preservation_weight"] != 1
            or config["checkpoint_selection"] != "fixed final epoch; no development scene or holdout selection"):
        raise ValueError("Invalid bounded refinement configuration")


def refinement_loss(temperature, quality_scale):
    import tensorflow as tf

    def loss(target, output):
        mentor = target[:, :3] / temperature
        pupil = output[:, :3] / temperature
        preservation = tf.reduce_sum(tf.nn.softmax(mentor) *
            (tf.nn.log_softmax(mentor)-tf.nn.log_softmax(pupil)), axis=-1) * temperature**2
        quality = target[:, 4] * tf.square(output[:, 3]-target[:, 3]) * quality_scale
        return preservation + quality

    return loss


def train(config_path, student_run, mode, output):
    config = read_json(config_path)
    check_config(config, mode)
    output, run = Path(output), Path(student_run)
    if output.exists():
        raise FileExistsError(output)
    config_hash = sha256(config_path)
    qconfig = read_json(config["quality_config"])
    rows, receipts = training_rows(qconfig)
    meta = read_json(run / "student.fp32.json")
    checkpoint = run / "student.keras"
    if (meta.get("source_student_sha256") != sha256(checkpoint)
            or meta["sha256"] != sha256(run / "student.fp32.tflite")
            or meta["classes"] != ["background", "red_balloon", "blue_balloon"]
            or meta["size"] != 64 or meta["task"] != "balloon" or meta["synthetic_training"]
            or meta["training"]["manifest_sha256"] != sha256(config["classification_manifest"])):
        raise ValueError("Parent student, classification manifest or taxonomy identity changed")
    class_doc = checked_base(config["classification_manifest"], meta["training"]["config"])
    allowed_sources = {r["source_image"] for r in class_doc["samples"] if r["split"] == "train"}
    if any(r["source_image"] not in allowed_sources for r in rows):
        raise ValueError("Quality source not in original student training partition")
    class_rows, qtrain, qval = split_refinement(class_doc["samples"], rows, config["quality_holdout_fold"])
    class_root = Path(config["classification_manifest"]).resolve().parent
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    seed = meta["training"]["config"]["seed"]
    keras.utils.set_random_seed(seed)
    pupil = keras.models.load_model(checkpoint, compile=False)
    initial_weights = [w.copy() for w in pupil.get_weights()]
    classifier_weights = [w.copy() for w in pupil.get_layer("classifier").get_weights()]
    qx = np.stack([load_rgb(r["absolute_path"], 64) for r in qtrain])
    qy = np.array([r["quality"] for r in qtrain], np.float32)
    cx = np.stack([load_rgb(class_root / r["path"], 64) for r in class_rows])
    x = np.concatenate([cx, qx])
    parent_logits = pupil(x, training=False).numpy()
    features_model = keras.Model(pupil.input, pupil.get_layer("spatial_features").output)
    w, bias = fit_ridge(features_model(qx, training=False).numpy(), qy, qconfig["ridge_penalty"])
    pupil.trainable = mode == "joint"
    pupil.get_layer("classifier").trainable = False
    head = keras.layers.Dense(1, name="box_quality_raw")
    q = head(pupil.get_layer("spatial_features").output)
    model = keras.Model(pupil.input, keras.layers.Concatenate(name="class_logits_and_quality")([pupil.output, q]))
    head.set_weights([w[:, None], np.array([bias], np.float32)])
    target = np.concatenate([parent_logits,
        np.concatenate([np.zeros(len(cx)), qy])[:, None],
        np.concatenate([np.zeros(len(cx)), np.ones(len(qx))])[:, None]], axis=1).astype(np.float32)
    initial = model(x, training=False).numpy()
    output.mkdir(parents=True)
    provenance = dict(config=config, config_sha256=config_hash, mode=mode, seed=seed,
        source_student_sha256=sha256(checkpoint), source_tflite_sha256=meta["sha256"],
        quality_config_sha256=sha256(config["quality_config"]), source_reviews=receipts,
        classification_manifest_sha256=sha256(config["classification_manifest"]),
        classification_training_entries=len(cx), quality_training_entries=len(qx),
        quality_holdout_entries=len(qval), heldout_source_groups=sorted({r["source_group"] for r in qval}),
        script_sha256=sha256(__file__), test_evaluated=False, deployment_approved=False,
        heldout_caveat="Excluded from current refinement; original backbone has seen related imagery",
        quality_guard_sha256=sha256("scripts/train_balloon_quality.py"))
    write_json(output / "provenance.json", provenance)
    write_json(output / "status.json", dict(state="running"))
    try:
        ds = tf.data.Dataset.from_tensor_slices((x, target)).shuffle(len(x), seed=seed).batch(config["batch_size"])
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        ds = ds.with_options(options).prefetch(1)
        model.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]),
                      loss=refinement_loss(config["temperature"], len(x)/len(qx)))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if not logs or not all(np.isfinite(v) for v in logs.values()):
                    raise ValueError("Non-finite joint training metrics")
                write_json(output / "status.json", dict(state="running", epoch=epoch+1, metrics=logs))

        model.fit(ds, epochs=config["epochs"], verbose=2,
                  callbacks=[Progress(), keras.callbacks.CSVLogger(str(output / "epochs.csv"))])
        if any(not np.array_equal(a, b) for a, b in zip(classifier_weights, pupil.get_layer("classifier").get_weights())):
            raise ValueError("Frozen classifier weights changed")
        if mode == "head_only" and any(not np.array_equal(a, b) for a, b in zip(initial_weights, pupil.get_weights())):
            raise ValueError("Head-only control changed its backbone")
        final = model(x, training=False).numpy()
        hold_x = np.stack([load_rgb(r["absolute_path"], 64) for r in qval])
        hold_y = np.array([r["quality"] for r in qval])
        hold_pred = np.clip(model(hold_x, training=False).numpy()[:, 3], 0, 1)
        val_rows = [r for r in class_doc["samples"] if r["split"] == "val"]
        vx = np.stack([load_rgb(class_root/r["path"], 64) for r in val_rows])
        val_classes = model(vx, training=False).numpy()[:, :3].argmax(axis=1)
        model.save(output / "quality.keras")

        @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
        def serve(inputs):
            return model(inputs, training=False)

        frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
        binary = tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()
        path = output / "quality.fp32.tflite"
        path.write_bytes(binary)
        interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        delta = 0.
        for crop, expected in zip(x, final):
            interpreter.set_tensor(inp["index"], crop[None])
            interpreter.invoke()
            delta = max(delta, float(np.abs(interpreter.get_tensor(out["index"])[0]-expected).max()))
        if delta > .002 or not np.isfinite(final).all():
            raise ValueError("Export parity or finiteness failed")
        if sha256(checkpoint) != provenance["source_student_sha256"] or sha256(config_path) != config_hash:
            raise ValueError("Refinement inputs changed")
        metadata = dict(kind="balloon_quality_research", classes=meta["classes"], size=64, threshold=.8,
            output="first 3: class logits; fourth: raw IoU regression, clip to [0,1]", tensor_dtype="float32",
            sha256=sha256(path), bytes=len(binary), source_student_sha256=provenance["source_student_sha256"],
            source_tflite_sha256=meta["sha256"], training_mode=mode, parameters=model.count_params(),
            added_parameters=head.count_params(), training=provenance, test_evaluated=False,
            synthetic_training=False, deployment_approved=False, pi_zero_verified=False)
        write_json(path.with_suffix(".json"), metadata)
        report = dict(mode=mode, initial_training_quality_mae=float(np.abs(np.clip(initial[len(cx):, 3], 0, 1)-qy).mean()),
            final_training_quality_mae=float(np.abs(np.clip(final[len(cx):, 3], 0, 1)-qy).mean()),
            quality_label_holdout_mae=float(np.abs(hold_pred-hold_y).mean()),
            class_probability_drift_max=float(np.abs(softmax(final[:, :3])-softmax(parent_logits)).max()),
            class_argmax_changes_training=int(np.sum(final[:, :3].argmax(1) != parent_logits.argmax(1))),
            validation_crop_correct=int(sum(int(a) == meta["classes"].index(r["label"]) for a, r in zip(val_classes, val_rows))),
            validation_crop_count=len(val_rows), max_keras_tflite_delta=delta,
            source_classifier_weights_unchanged=True, backbone_weights_unchanged=mode == "head_only",
            holdout_examples=[dict(source_group=r["source_group"], target=float(y), prediction=float(p))
                              for r, y, p in zip(qval, hold_y, hold_pred)],
            test_evaluated=False, deployment_approved=False, pi_timing_measured=False)
        write_json(output / "report.json", report)
        write_json(output / "status.json", dict(state="complete"))
        print({k: v for k, v in report.items() if k != "holdout_examples"})
    except BaseException as exc:
        write_json(output / "status.json", dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "student-run", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--mode", choices=("joint", "head_only"), required=True)
    args = parser.parse_args()
    train(args.config, args.student_run, args.mode, args.output)
