"""Teacher-only training with durable epoch-boundary resume and explicit evaluation."""

import fcntl
import json
import math
import platform
import time
import uuid
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .data import batches, read_json, sha256, validate
from .models import freeze_backbone, load_backbone, teacher

PHASES = (
    ("head", "head_epochs", "head_lr", 1e-3),
    ("finetune", "finetune_epochs", "finetune_lr", 1e-5),
)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reference(path, root):
    return {"path": str(Path(path).relative_to(root)), "sha256": sha256(path)}


def checked_checkpoint(root, item):
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != item["sha256"]:
        raise ValueError("Checkpoint path/checksum mismatch")
    return path


class EpochCheckpoint(keras.callbacks.Callback):
    """Commit model+optimizer first, then atomically point state to that checkpoint."""

    def __init__(self, root, state, config, budget=None):
        super().__init__()
        self.root, self.state, self.config, self.budget = root, state, config, budget
        self.used = 0

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        if "val_loss" not in metrics or not all(math.isfinite(x) for x in metrics.values()):
            raise ValueError("Non-finite or missing epoch metrics; checkpoint not committed")
        phase = PHASES[self.state["phase"]][0]
        path = self.root / "checkpoints" / f"{phase}-{epoch + 1:03d}-{uuid.uuid4().hex[:8]}.keras"
        temporary = path.with_name(path.stem + ".pending.keras")
        self.model.save(temporary)
        temporary.replace(path)
        latest = reference(path, self.root)
        best = self.state["best"]
        if best is None or metrics["val_loss"] < best["val_loss"] - self.config.get(
            "min_delta", 0.0
        ):
            self.state.update(
                best={**latest, "val_loss": metrics["val_loss"], "epoch": epoch + 1}, wait=0
            )
        else:
            self.state["wait"] += 1
        self.state.update(latest=latest, completed=epoch + 1)
        self.state["history"].append({"phase": phase, "epoch": epoch + 1, **metrics})
        atomic_json(self.root / "state.json", self.state)
        atomic_json(
            self.root / "status.json", {"state": "running", "stage": phase, "epoch": epoch + 1}
        )
        self.used += 1
        if self.state["wait"] >= self.config.get("patience", 5) or (
            self.budget is not None and self.used >= self.budget
        ):
            self.model.stop_training = True


def effective_config(config, smoke):
    config = dict(config)
    if smoke:
        config.update(head_epochs=1, finetune_epochs=1, batch_size=8)
    for key in ("head_epochs", "finetune_epochs", "batch_size", "teacher_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("patience", 5) < 1 or config.get("min_delta", 0) < 0:
        raise ValueError("Invalid early-stopping settings")
    for _, _, key, default in PHASES:
        if not math.isfinite(config.get(key, default)) or config.get(key, default) <= 0:
            raise ValueError(f"Invalid {key}")
    return config


def train_teacher(
    manifest,
    config,
    output,
    pretrained=None,
    smoke=False,
    resume=False,
    epoch_budget=None,
    initial_teacher=None,
):
    config = effective_config(config, smoke)
    if epoch_budget is not None and epoch_budget < 1:
        raise ValueError("Epoch budget must be positive")
    doc = validate(manifest, config)
    if doc.get("training_approved") is False:
        raise ValueError("Dataset is quarantined pending label review")
    if doc["synthetic"] and not smoke:
        raise ValueError("Synthetic fixtures require --smoke; not real DTR training")
    root = Path(output).resolve()
    if resume:
        state = read_json(root / "state.json")
        if read_json(root / "config.json") != config:
            raise ValueError("Resume config differs from original run")
        provenance = read_json(root / "provenance.json")
        if provenance["manifest_sha256"] != sha256(manifest) or provenance["smoke"] != smoke:
            raise ValueError("Resume dataset or smoke mode differs from original run")
        if provenance["tensorflow"] != tf.__version__ or provenance["keras"] != keras.__version__:
            raise ValueError("Resume requires the original TensorFlow/Keras versions")
        if (root / "report.json").exists():
            raise ValueError("Teacher run is complete; use a new run for a new experiment")
    else:
        if root.exists():
            raise FileExistsError(f"Use a fresh teacher run directory: {root}")
        if pretrained and initial_teacher:
            raise ValueError("Choose pretrained backbone OR initial teacher, not both")
        source_checkpoint = initial_teacher or pretrained
        if not source_checkpoint:
            raise ValueError("Teacher training requires a verified pretrained Keras checkpoint")
        receipt = read_json(Path(source_checkpoint).with_suffix(".json"))
        if receipt.get("sha256") != sha256(source_checkpoint):
            raise ValueError("Initial checkpoint checksum mismatch")
        if initial_teacher:
            if (
                receipt.get("kind") != "keras_teacher"
                or receipt.get("task") != config["task"]
                or receipt.get("classes") != config["classes"]
                or receipt.get("size") != config["teacher_size"]
            ):
                raise ValueError("Initial teacher task/classes/size metadata mismatch")
            if receipt.get("synthetic_training", True) and not smoke:
                raise ValueError("Synthetic initial teacher requires --smoke")
        elif not receipt.get("parity_passed"):
            raise ValueError("Pretrained checkpoint lacks valid parity/checksum receipt")
        root.mkdir(parents=True)
        (root / "checkpoints").mkdir()
        provenance = {
            "kind": "teacher_only",
            "manifest": str(Path(manifest).resolve()),
            "manifest_sha256": sha256(manifest),
            "synthetic": doc["synthetic"],
            "smoke": smoke,
            "split_provenance": doc.get("split_provenance"),
            "dataset_qualification": doc.get("qualification", {}),
            "task": config["task"],
            "pretrained": str(Path(source_checkpoint).resolve()),
            "pretrained_sha256": sha256(source_checkpoint),
            "initialization": "complete_teacher" if initial_teacher else "imagenet_backbone",
            "initial_metadata_sha256": sha256(Path(source_checkpoint).with_suffix(".json")),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "machine": platform.machine(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resume_semantics": "committed epoch boundary; optimizer restored; RNG stream not restored",
        }
        state = dict(
            phase=0,
            completed=0,
            latest=None,
            best=None,
            wait=0,
            phase_input=None,
            candidates=[],
            history=[],
        )
        atomic_json(root / "config.json", config)
        atomic_json(root / "provenance.json", provenance)
        atomic_json(root / "state.json", state)
    with (root / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This teacher run is already being trained") from exc
        # Read committed state after acquiring the lock, not a potentially stale snapshot.
        state = read_json(root / "state.json")
        if (root / "report.json").exists():
            raise ValueError("Teacher run is already complete")
        keras.utils.set_random_seed(config["seed"])
        try:
            remaining = epoch_budget
            while state["phase"] < len(PHASES):
                phase, epoch_key, lr_key, default_lr = PHASES[state["phase"]]
                done = state["completed"] >= config[epoch_key] or state["wait"] >= config.get(
                    "patience", 5
                )
                if done:
                    state["candidates"].append({"phase": phase, **state["best"]})
                    state.update(
                        phase=state["phase"] + 1,
                        completed=0,
                        latest=None,
                        phase_input=state["best"],
                        best=None,
                        wait=0,
                    )
                    atomic_json(root / "state.json", state)
                    continue
                if remaining == 0:
                    result = {
                        "state": "paused",
                        "stage": phase,
                        "reason": "epoch budget reached",
                        "synthetic_training": doc["synthetic"],
                        "distillation_started": False,
                    }
                    atomic_json(root / "status.json", result)
                    return result
                if state["latest"]:
                    model = keras.models.load_model(checked_checkpoint(root, state["latest"]))
                else:
                    if state["phase_input"]:
                        model = keras.models.load_model(
                            checked_checkpoint(root, state["phase_input"]), compile=False
                        )
                    else:
                        source = Path(provenance["pretrained"])
                        if sha256(source) != provenance["pretrained_sha256"]:
                            raise ValueError("Pretrained source changed")
                        if provenance.get("initialization") == "complete_teacher":
                            model = keras.models.load_model(source, compile=False)
                            if model.input_shape != (
                                None,
                                config["teacher_size"],
                                config["teacher_size"],
                                3,
                            ) or model.output_shape != (None, len(config["classes"])):
                                raise ValueError("Initial teacher tensor shape mismatch")
                        else:
                            model = teacher(len(config["classes"]), config["teacher_size"])
                            load_backbone(model, keras.models.load_model(source, compile=False))
                    freeze_backbone(model, phase == "head")
                    model.compile(
                        optimizer=keras.optimizers.Adam(config.get(lr_key, default_lr)),
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=["accuracy"],
                    )
                folder = root / phase
                folder.mkdir(exist_ok=True)
                checkpoint = EpochCheckpoint(root, state, config, remaining)
                atomic_json(root / "status.json", {"state": "running", "stage": phase})
                model.fit(
                    batches(
                        manifest,
                        config,
                        "train",
                        config["teacher_size"],
                        shuffle=True,
                        augment=True,
                    ),
                    validation_data=batches(manifest, config, "val", config["teacher_size"]),
                    initial_epoch=state["completed"],
                    epochs=config[epoch_key],
                    verbose=2,
                    callbacks=[
                        checkpoint,
                        keras.callbacks.CSVLogger(str(folder / "epochs.csv"), append=True),
                        keras.callbacks.TensorBoard(log_dir=str(folder / "tensorboard")),
                    ],
                )
                if remaining is not None:
                    remaining -= checkpoint.used
            selected = min(state["candidates"], key=lambda x: x["val_loss"])
            model = keras.models.load_model(checked_checkpoint(root, selected), compile=False)
            temporary = root / "teacher.pending.keras"
            model.save(temporary)
            temporary.replace(root / "teacher.keras")
            metadata = {
                "kind": "keras_teacher",
                "task": config["task"],
                "classes": config["classes"],
                "size": config["teacher_size"],
                "threshold": config["threshold"],
                "color": "RGB",
                "resize": "Pillow bilinear stretch",
                "pixel_range": [0, 255],
                "normalization": "embedded ImageNet normalization",
                "output": "logits",
                "synthetic_training": doc["synthetic"],
                "deployment_approved": False,
                "sha256": sha256(root / "teacher.keras"),
                "dataset_qualification": doc.get("qualification", {}),
                "selected_checkpoint": selected,
                "manifest_sha256": provenance["manifest_sha256"],
                "distillation_started": False,
            }
            atomic_json(root / "teacher.json", metadata)
            # No test-set predictions during model selection.
            metrics = evaluate_teacher(root, manifest, split="val")
            atomic_json(root / "validation.json", metrics)
            report = {
                "status": "TEACHER_SMOKE_PASSED" if smoke else "TEACHER_TRAINED_UNVALIDATED",
                "dataset_qualification": doc.get("qualification", {}),
                "synthetic_training": doc["synthetic"],
                "deployment_approved": False,
                "distillation_started": False,
                "test_evaluated": False,
                "selected_checkpoint": selected,
                "validation": metrics,
                "history": state["history"],
            }
            atomic_json(root / "report.json", report)
            atomic_json(root / "status.json", {"state": "complete", "result": report["status"]})
            return report
        except BaseException as exc:
            atomic_json(
                root / "status.json",
                {
                    "state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "error": str(exc),
                    "resume": "--resume restores the last committed epoch",
                },
            )
            raise


def classification_report(actual, scores, classes, threshold):
    actual, scores = np.asarray(actual), np.asarray(scores)
    predicted = scores.argmax(axis=1)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(confusion, (actual, predicted), 1)
    tp = confusion.diagonal()
    precision = np.divide(
        tp, confusion.sum(0), out=np.zeros(len(classes)), where=confusion.sum(0) > 0
    )
    recall = np.divide(tp, confusion.sum(1), out=np.zeros(len(classes)), where=confusion.sum(1) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes)),
        where=(precision + recall) > 0,
    )
    bg = classes.index("background")
    accepted = (predicted != bg) & (scores.max(1) >= threshold)
    correct = accepted & (predicted == actual)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "macro_f1": float(f1.mean()),
        "classes": classes,
        "samples": len(actual),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": confusion.sum(1).tolist(),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "threshold": threshold,
        "accepted_count": int(accepted.sum()),
        "wrong_accepted_count": int((accepted & ~correct).sum()),
        "accepted_precision": float(correct.sum() / accepted.sum()) if accepted.any() else None,
        "correct_accepted_target_recall": float(correct.sum() / max(1, (actual != bg).sum())),
        "background_accept_rate": float(
            (accepted & (actual == bg)).sum() / max(1, (actual == bg).sum())
        ),
    }


def evaluate_teacher(run, manifest=None, split="val", output=None):
    from .teacher_runtime import TeacherPredictor

    root = Path(run).resolve()
    config = read_json(root / "config.json")
    provenance = read_json(root / "provenance.json")
    manifest = manifest or provenance["manifest"]
    doc = validate(manifest, config)
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if sha256(manifest) != provenance["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the recorded training dataset")
    if output and Path(output).exists():
        raise FileExistsError(output)
    predictor = TeacherPredictor(root / "teacher.keras", allow_unvalidated=True)
    rows = [r for r in doc["samples"] if r["split"] == split]
    predictions = [predictor.predict(Path(manifest).resolve().parent / r["path"]) for r in rows]
    actual = [config["classes"].index(r["label"]) for r in rows]
    scores = np.asarray([p["scores"] for p in predictions])
    report = {
        "split": split,
        "dataset_qualification": doc.get("qualification", {}),
        "synthetic_training": doc["synthetic"],
        "model_sha256": predictor.metadata["sha256"],
        "manifest_sha256": sha256(manifest),
        "deployment_approved": False,
        "metrics": classification_report(actual, scores, config["classes"], config["threshold"]),
        "predictions": [
            {"path": r["path"], "session": r["session"], "actual": r["label"], **p}
            for r, p in zip(rows, predictions)
        ],
    }
    if split == "val":
        report["threshold_sweep"] = [
            classification_report(actual, scores, config["classes"], t)
            for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
        ]
    if output:
        dest = Path(output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
