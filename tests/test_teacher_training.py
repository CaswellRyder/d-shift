from pathlib import Path

import keras
import numpy as np
import pytest

from dtr.data import read_json, sha256, synthetic, write_json
from dtr.teacher_runtime import TeacherPredictor
from dtr.teacher_training import classification_report, evaluate_teacher, train_teacher

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # Tiny native Keras model for lifecycle tests. Real V4 is tested in CLI smoke runs.
    def tiny(classes, size):
        inputs = keras.Input((size, size, 3))
        x = keras.layers.Rescaling(1 / 255)(inputs)
        x = keras.layers.Conv2D(4, 3, strides=4, name="features")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        return keras.Model(inputs, keras.layers.Dense(classes, name="classifier")(x))

    monkeypatch.setattr("dtr.teacher_training.teacher", tiny)
    config = read_json(ROOT / "configs/balloon.json")
    config.update(head_epochs=2, finetune_epochs=1, batch_size=3)
    manifest = synthetic(tmp_path / "data", config, per_class=2)
    # Explicit fixture-only non-synthetic marker to exercise the full epoch schedule.
    doc = read_json(manifest)
    doc["synthetic"] = False
    doc["source"] = "unit-test fixture ONLY"
    doc["qualification"] = {"domain": "unit-test ONLY", "competition_accuracy_established": False}
    write_json(manifest, doc)
    pretrained = tmp_path / "pretrained.keras"
    tiny(3, config["teacher_size"]).save(pretrained)
    write_json(
        pretrained.with_suffix(".json"), {"parity_passed": True, "sha256": sha256(pretrained)}
    )
    return config, manifest, pretrained, tmp_path / "run"


def test_teacher_resume_optimizer_validation_and_explicit_test(setup):
    config, manifest, pretrained, run = setup
    paused = train_teacher(manifest, config, run, pretrained, epoch_budget=1)
    assert paused["state"] == "paused" and not paused["distillation_started"]
    state = read_json(run / "state.json")
    assert state["completed"] == 1 and state["phase"] == 0
    first = keras.models.load_model(run / state["latest"]["path"])
    iterations = int(first.optimizer.iterations.numpy())
    assert iterations > 0
    unchanged = (run / "status.json").read_text()
    with pytest.raises(ValueError, match="config differs"):
        train_teacher(manifest, {**config, "teacher_size": 64}, run, resume=True)
    assert (run / "status.json").read_text() == unchanged
    train_teacher(manifest, config, run, resume=True, epoch_budget=1)
    second = keras.models.load_model(next((run / "checkpoints").glob("head-002-*.keras")))
    assert int(second.optimizer.iterations.numpy()) == 2 * iterations
    report = train_teacher(manifest, config, run, resume=True)
    assert not report["distillation_started"] and not report["test_evaluated"]
    assert report["dataset_qualification"]["domain"] == "unit-test ONLY"
    assert (
        read_json(run / "teacher.json")["dataset_qualification"] == report["dataset_qualification"]
    )
    assert (
        read_json(run / "provenance.json")["dataset_qualification"]
        == report["dataset_qualification"]
    )
    assert report["validation"]["split"] == "val"
    assert read_json(run / "validation.json") == report["validation"]
    assert not list(run.glob("student*"))
    assert len(report["history"]) == 3
    assert (run / "teacher.keras").exists()
    with pytest.raises(ValueError, match="Unvalidated"):
        TeacherPredictor(run / "teacher.keras")
    predictor = TeacherPredictor(run / "teacher.keras", True)
    assert len(predictor.predict(np.zeros((30, 40, 3), np.uint8))["scores"]) == 3
    crop = np.zeros((30, 40, 3), np.uint8)
    single = predictor.predict(crop)
    batched = predictor.predict_many([crop, crop])
    np.testing.assert_allclose(single["scores"], batched[0]["scores"], atol=1e-6)
    assert len(batched) == 2 and predictor.predict_many([]) == []
    with pytest.raises(ValueError, match="64"):
        predictor.predict_many([crop] * 65)
    test_report = evaluate_teacher(run, split="test", output=run / "test-evaluation.json")
    assert test_report["split"] == "test" and "threshold_sweep" not in test_report
    assert test_report["metrics"]["samples"] == 6
    with pytest.raises(FileExistsError):
        evaluate_teacher(run, output=run / "test-evaluation.json")
    with pytest.raises(ValueError, match="complete"):
        train_teacher(manifest, config, run, resume=True)


def test_teacher_interruption_before_first_epoch_is_resumable(setup, monkeypatch):
    config, manifest, pretrained, run = setup
    with monkeypatch.context() as patch:

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        patch.setattr(keras.Model, "fit", interrupt)
        with pytest.raises(KeyboardInterrupt):
            train_teacher(manifest, config, run, pretrained)
    assert read_json(run / "status.json")["state"] == "interrupted"
    assert read_json(run / "state.json")["completed"] == 0
    result = train_teacher(manifest, config, run, resume=True, epoch_budget=1)
    assert result["state"] == "paused"


def test_synthetic_training_requires_opt_in(setup):
    config, manifest, pretrained, run = setup
    doc = read_json(manifest)
    doc["synthetic"] = True
    write_json(manifest, doc)
    with pytest.raises(ValueError, match="--smoke"):
        train_teacher(manifest, config, run, pretrained)
    assert not run.exists()


def test_quarantined_labels_cannot_train(setup):
    from dtr.data import batches
    from dtr.training import run as legacy_train

    config, manifest, pretrained, run = setup
    doc = read_json(manifest)
    doc["training_approved"] = False
    write_json(manifest, doc)
    with pytest.raises(ValueError, match="quarantined"):
        train_teacher(manifest, config, run, pretrained)
    with pytest.raises(ValueError, match="quarantined"):
        legacy_train(manifest, config, run, pretrained)
    with pytest.raises(ValueError, match="quarantined"):
        batches(manifest, config, "train", config["teacher_size"])
    assert not run.exists()


def test_changed_manifest_rejected_on_resume(setup):
    config, manifest, pretrained, run = setup
    train_teacher(manifest, config, run, pretrained, epoch_budget=1)
    doc = read_json(manifest)
    doc["source"] = "changed data provenance"
    write_json(manifest, doc)
    with pytest.raises(ValueError, match="dataset"):
        train_teacher(manifest, config, run, resume=True)


def test_acceptance_metrics_do_not_hide_rejections():
    report = classification_report(
        [0, 1, 2],
        [[0.1, 0.85, 0.05], [0.1, 0.6, 0.3], [0.05, 0.05, 0.9]],
        ["background", "green", "purple"],
        0.8,
    )
    assert report["accepted_count"] == 2
    assert report["wrong_accepted_count"] == 1
    assert report["accepted_precision"] == 0.5
    assert report["correct_accepted_target_recall"] == 0.5
    assert report["background_accept_rate"] == 1
    no_accept = classification_report(
        [0, 1], [[0.5, 0.5], [0.5, 0.5]], ["background", "green"], 0.9
    )
    assert no_accept["accepted_precision"] is None


def test_verified_warm_start_retains_classifier_and_rejects_mismatch(setup, monkeypatch):
    config, manifest, pretrained, run = setup
    train_teacher(manifest, config, run, pretrained)
    source = run / "teacher.keras"
    original = keras.models.load_model(source, compile=False)
    captured = []

    def inspect_then_interrupt(model, *args, **kwargs):
        captured.extend(model.get_layer("classifier").get_weights())
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(keras.Model, "fit", inspect_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            train_teacher(manifest, config, run.parent / "warm", initial_teacher=source)
    for expected, actual in zip(original.get_layer("classifier").get_weights(), captured):
        np.testing.assert_array_equal(expected, actual)
    assert read_json(run.parent / "warm/provenance.json")["initialization"] == "complete_teacher"
    result = train_teacher(manifest, config, run.parent / "warm", resume=True, epoch_budget=1)
    assert result["state"] == "paused"
    metadata = read_json(source.with_suffix(".json"))
    metadata["task"] = "goal"
    write_json(source.with_suffix(".json"), metadata)
    with pytest.raises(ValueError, match="metadata mismatch"):
        train_teacher(manifest, config, run.parent / "wrong", initial_teacher=source)
    assert not (run.parent / "wrong").exists()
