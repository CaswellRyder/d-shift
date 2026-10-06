"""Keras teacher fine-tuning, aligned crop distillation, and held-out evaluation."""

import platform
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .data import batches, read_json, sha256, validate, write_json
from .export import export_int8
from .models import freeze_backbone, load_backbone, student, teacher


class Distiller(keras.Model):
    def __init__(self, pupil, mentor, config):
        super().__init__()
        self.pupil, self.mentor = pupil, mentor
        self.mentor.trainable = False
        self.temperature, self.alpha = config["temperature"], config["alpha"]
        self.size = config["student_size"]
        if self.temperature <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError("Invalid distillation temperature or alpha")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy]

    def call(self, x, training=False):
        return self.pupil(x["student"], training=training)

    def train_step(self, data):
        x, y = data
        target = tf.stop_gradient(self.mentor(x["teacher"], training=False))
        with tf.GradientTape() as tape:
            predicted = self(x, training=True)
            hard = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y, predicted, from_logits=True)
            )
            t = self.temperature
            p = tf.nn.softmax(target / t)
            log_p = tf.nn.log_softmax(target / t)
            log_q = tf.nn.log_softmax(predicted / t)
            soft = tf.reduce_mean(tf.reduce_sum(p * (log_p - log_q), axis=-1)) * t**2
            loss = self.alpha * hard + (1 - self.alpha) * soft
        grads = tape.gradient(loss, self.pupil.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.pupil.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, predicted)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        predicted = self(x, training=False)
        loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(y, predicted, from_logits=True)
        )
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, predicted)
        return {m.name: m.result() for m in self.metrics}


def evaluate(model, ds, classes):
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for x, y in ds:
        predicted = model(x, training=False).numpy().argmax(axis=-1)
        for actual, guess in zip(y.numpy(), predicted):
            confusion[int(actual), int(guess)] += 1
    precision = np.divide(
        confusion.diagonal(),
        confusion.sum(axis=0),
        out=np.zeros(len(classes), dtype=float),
        where=confusion.sum(axis=0) > 0,
    )
    recall = np.divide(
        confusion.diagonal(),
        confusion.sum(axis=1),
        out=np.zeros(len(classes), dtype=float),
        where=confusion.sum(axis=1) > 0,
    )
    return {
        "accuracy": float(confusion.trace() / max(1, confusion.sum())),
        "classes": classes,
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "samples": int(confusion.sum()),
    }


def callbacks(folder):
    return [
        keras.callbacks.CSVLogger(str(folder / "epochs.csv")),
        keras.callbacks.TensorBoard(log_dir=str(folder / "tensorboard")),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.TerminateOnNaN(),
    ]


def run(manifest, config, output, pretrained, smoke=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory: {output}")
    doc = validate(manifest, config)
    if doc.get("training_approved") is False:
        raise ValueError("Dataset is quarantined pending label review")
    if doc["synthetic"] and not smoke:
        raise ValueError("Synthetic data only allowed with --smoke; not a real training run")
    if not smoke and not pretrained:
        raise ValueError("Real training requires a verified pretrained Keras checkpoint")
    keras.utils.set_random_seed(config["seed"])
    output.mkdir(parents=True)
    config = dict(config)
    if smoke:
        config.update(head_epochs=1, finetune_epochs=1, student_epochs=1, batch_size=8)
    provenance = {
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": sha256(manifest),
        "synthetic": doc["synthetic"],
        "smoke": smoke,
        "split_provenance": doc.get("split_provenance"),
        "host": platform.node(),
        "machine": platform.machine(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pretrained": str(pretrained) if pretrained else None,
        "pretrained_sha256": sha256(pretrained) if pretrained else None,
    }
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", provenance)
    write_json(output / "status.json", {"state": "running", "stage": "teacher"})
    mentor = teacher(len(config["classes"]), config["teacher_size"])
    if pretrained:
        receipt = read_json(Path(pretrained).with_suffix(".json"))
        if not receipt.get("parity_passed") or receipt["sha256"] != sha256(pretrained):
            raise ValueError("Pretrained import lacks valid numerical-parity receipt")
        load_backbone(mentor, keras.models.load_model(pretrained, compile=False))
    size = config["teacher_size"]
    train = batches(manifest, config, "train", size, shuffle=True, augment=True)
    val = batches(manifest, config, "val", size)
    for phase, epochs, lr in [
        ("head", config["head_epochs"], 1e-3),
        ("finetune", config["finetune_epochs"], 1e-5),
    ]:
        folder = output / phase
        folder.mkdir()
        freeze_backbone(mentor, phase == "head")
        mentor.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        mentor.fit(
            train, validation_data=val, epochs=epochs, callbacks=callbacks(folder), verbose=2
        )
    mentor.save(output / "teacher.keras")
    teacher_metrics = evaluate(mentor, batches(manifest, config, "test", size), config["classes"])
    write_json(output / "status.json", {"state": "running", "stage": "distillation"})
    pupil = student(len(config["classes"]), config["student_size"])
    distiller = Distiller(pupil, mentor, config)
    distiller.compile(optimizer=keras.optimizers.Adam(1e-3))
    folder = output / "distillation"
    folder.mkdir()
    paired_train = batches(manifest, config, "train", size, shuffle=True, augment=True, paired=True)
    paired_val = batches(manifest, config, "val", size, paired=True)
    distiller.fit(
        paired_train,
        validation_data=paired_val,
        epochs=config["student_epochs"],
        callbacks=callbacks(folder),
        verbose=2,
    )
    pupil.save(output / "student.keras")
    pupil_metrics = evaluate(
        pupil, batches(manifest, config, "test", config["student_size"]), config["classes"]
    )
    write_json(output / "status.json", {"state": "running", "stage": "int8_export"})
    exported = export_int8(pupil, manifest, config, output / "student.int8.tflite", provenance)
    from .runtime import Predictor, benchmark

    predictor = Predictor(output / "student.int8.tflite", allow_unvalidated=True)
    confusion = np.zeros((len(config["classes"]), len(config["classes"])), dtype=int)
    root = Path(manifest).resolve().parent
    test_rows = [r for r in doc["samples"] if r["split"] == "test"]
    for row in test_rows:
        result = predictor.predict(root / row["path"])
        confusion[
            config["classes"].index(row["label"]), config["classes"].index(result["label"])
        ] += 1
    report = {
        "status": "PIPELINE_SMOKE_PASSED" if smoke else "TRAINING_COMPLETE_UNVALIDATED",
        "synthetic_training": doc["synthetic"],
        "deployment_approved": False,
        "teacher": teacher_metrics,
        "student_fp32": pupil_metrics,
        "student_int8": {
            "accuracy": float(confusion.trace() / confusion.sum()),
            "confusion_true_rows_predicted_columns": confusion.tolist(),
        },
        "export": exported,
        "benchmark": benchmark(
            output / "student.int8.tflite",
            root / test_rows[0]["path"],
            30,
            allow_unvalidated=True,
        ),
    }
    report["quantization_accuracy_delta"] = (
        report["student_int8"]["accuracy"] - pupil_metrics["accuracy"]
    )
    write_json(output / "report.json", report)
    write_json(output / "status.json", {"state": "complete", "result": report["status"]})
    return report
