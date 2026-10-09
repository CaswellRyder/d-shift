import numpy as np
import pytest

from dtr.data import sha256, write_json
from scripts.distill_balloon_features import (
    checked_targets,
    normalize_features,
    training_boundary,
    weights_hash,
)


def test_training_boundary_excludes_img_before_pixel_loading():
    doc = dict(
        training_approved=True,
        synthetic=False,
        samples=[
            dict(split="train", source_image="train/frame_000123_jpg.rf.test.jpg"),
            dict(split="test", source_image="test/IMG_reserved.jpg"),
        ],
    )
    assert len(training_boundary(doc)) == 1
    for source in ("train/IMG_1234_jpg.rf.test.jpg", "train/img_123_augmented.png", None):
        doc["samples"][0]["source_image"] = source
        with pytest.raises(ValueError, match="source|IMG"):
            training_boundary(doc)


def test_unit_features_are_per_example_and_finite():
    x = np.array([[3, 4], [0, 0], [6, 8]], np.float32)
    result = normalize_features(x)
    assert np.allclose(result, [[0.6, 0.8], [0, 0], [0.6, 0.8]])
    assert np.array_equal(x[0], [3, 4])
    for bad in ([], [[np.nan]], [1, 2], np.zeros((0, 1280))):
        with pytest.raises(ValueError):
            normalize_features(bad)


def test_cache_checks_hash_split_width_and_source_identity(tmp_path):
    features = normalize_features(np.ones((2, 1280)))
    path = tmp_path / "targets.npz"
    np.savez(path, logits=np.ones((2, 3)), features=features)
    receipt = dict(
        manifest_sha256="manifest",
        teacher_sha256="teacher",
        split="train",
        feature_layer="flatten",
        targets_sha256=sha256(path),
    )
    write_json(tmp_path / "receipt.json", receipt)
    a, b, _ = checked_targets(tmp_path, "manifest", "teacher", 2)
    assert a.shape == (2, 3) and b.shape == (2, 1280)
    with pytest.raises(ValueError, match="identity"):
        checked_targets(tmp_path, "different", "teacher", 2)
    with pytest.raises(ValueError, match="arrays"):
        checked_targets(tmp_path, "manifest", "teacher", 3)
    receipt["split"] = "val"
    write_json(tmp_path / "receipt.json", receipt)
    with pytest.raises(ValueError, match="identity"):
        checked_targets(tmp_path, "manifest", "teacher", 2)


def test_training_projection_is_not_part_of_exported_pupil():
    import keras
    from dtr.models import research_student

    keras.utils.set_random_seed(42)
    pupil = research_student(3, 64, "separable_context")
    identity = weights_hash(pupil)
    projection = keras.layers.Dense(1280)(pupil.get_layer("spatial_features").output)
    trainer = keras.Model(pupil.input, [pupil.output, projection])
    assert trainer.count_params() == 7763 + 165120
    assert pupil.count_params() == 7763 and pupil.output_shape == (None, 3)
    assert weights_hash(pupil) == identity
    image = np.zeros((1, 64, 64, 3), np.float32)
    assert np.array_equal(pupil(image).numpy(), trainer(image)[0].numpy())
