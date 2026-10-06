import importlib.util
from pathlib import Path

import keras
import numpy as np
import pytest

from dtr.data import write_json
from dtr.models import initialize_separable_context, research_student, student

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("distill_research", ROOT / "scripts/distill_pi_student.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


@pytest.mark.parametrize("variant,count", [("tiny", 6263), ("spatial", 6935),
                                          ("context", 16183), ("separable_context", 8279)])
def test_architecture_contract(variant, count, tmp_path):
    keras.utils.set_random_seed(42)
    model = research_student(7, 64, variant)
    assert model.count_params() == count
    assert model.input_shape == (None, 64, 64, 3)
    assert model.output_shape == (None, 7)
    if variant != "tiny":
        assert model.get_layer("spatial_pool").output.shape[1:] == (2, 2, 32)
    model.save(tmp_path / "model.keras")
    reloaded = keras.models.load_model(tmp_path / "model.keras", compile=False)
    x = np.ones((1, 64, 64, 3), np.float32)
    np.testing.assert_array_equal(model(x), reloaded(x))


def test_original_model_unchanged():
    keras.utils.set_random_seed(42)
    old = student(7)
    keras.utils.set_random_seed(42)
    new = research_student(7)
    for a, b in zip(old.get_weights(), new.get_weights()):
        np.testing.assert_array_equal(a, b)


def test_separable_warm_start_copies_stem_head_without_mutating_source():
    source = research_student(7, 64, "context")
    target = research_student(7, 64, "separable_context")
    before = [w.copy() for w in source.get_weights()]
    report = initialize_separable_context(source, target)
    assert 0 <= report["relative_kernel_error"] <= 1
    for old, new in zip(before, source.get_weights()):
        np.testing.assert_array_equal(old, new)
    for name in report["copied_layers"]:
        for a, b in zip(source.get_layer(name).get_weights(), target.get_layer(name).get_weights()):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="contract"):
        initialize_separable_context(source, research_student(4, 64, "separable_context"))


def test_rank_one_context_factorization_preserves_representable_network():
    source = research_student(7, 64, "context")
    target = research_student(7, 64, "separable_context")
    rng = np.random.default_rng(11)
    a = rng.normal(size=(3, 3, 32, 1)).astype(np.float32)*.03
    b = rng.normal(size=(1, 1, 32, 32)).astype(np.float32)*.03
    source.get_layer("context").set_weights([a*b, np.zeros(32, np.float32)])
    report = initialize_separable_context(source, target)
    assert report["relative_kernel_error"] < 1e-6
    x = rng.uniform(0, 255, (2, 64, 64, 3)).astype(np.float32)
    np.testing.assert_allclose(source(x), target(x), atol=1e-6)


@pytest.mark.parametrize("size,variant", [(31, "tiny"), (63, "spatial"), (64, "unknown")])
def test_bad_architecture_rejected(size, variant):
    with pytest.raises(ValueError):
        research_student(7, size, variant)


def test_cache_contract(tmp_path):
    classes, rows = ["background", "goal"], [{"label": "goal"}, {"label": "background"}]
    write_json(tmp_path / "provenance.json", dict(manifest_sha256="m", teacher_sha256="t",
                                               alpha=.5, config=dict(classes=classes)))
    targets = np.array([[0, 1, 3, 4], [1, 0, 5, 6]], dtype=np.float32)
    np.save(tmp_path / "train-targets.npy", targets)
    np.testing.assert_array_equal(TRAIN.checked_cache(tmp_path, "m", "t", rows, classes), targets)
    with pytest.raises(ValueError, match="identity"):
        TRAIN.checked_cache(tmp_path, "wrong", "t", rows, classes)
    with pytest.raises(ValueError, match="row labels"):
        TRAIN.checked_cache(tmp_path, "m", "t", rows[::-1], classes)
    targets[0, 2] = np.nan
    np.save(tmp_path / "train-targets.npy", targets)
    with pytest.raises(ValueError, match="Invalid"):
        TRAIN.checked_cache(tmp_path, "m", "t", rows, classes)


def test_supervised_placeholder_cache_cannot_distill(tmp_path):
    write_json(tmp_path / "provenance.json", dict(alpha=1, teacher_targets_kind="unused_zeros"))
    with pytest.raises(ValueError, match="verified teacher logits"):
        TRAIN.checked_cache(tmp_path, "m", "t", [], [])
