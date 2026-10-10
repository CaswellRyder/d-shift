import numpy as np
import pytest

from scripts.distill_pi_student import check_augmentation, geometric_crop


def test_seeded_symmetries_preserve_pixels_colors_shape_and_dtype():
    rgb = np.arange(8*8*3, dtype=np.float32).reshape(8, 8, 3)
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    outputs = []
    original_pixels = sorted(map(tuple, rgb.reshape(-1, 3)))
    for _ in range(64):
        result = geometric_crop(rgb, a)
        np.testing.assert_array_equal(result, geometric_crop(rgb, b))
        assert result.dtype == rgb.dtype and result.shape == rgb.shape
        assert result.flags.c_contiguous
        assert sorted(map(tuple, result.reshape(-1, 3))) == original_pixels
        outputs.append(result.tobytes())
    assert len(set(outputs)) == 8


@pytest.mark.parametrize("task,alpha,cache", [("goal", 1, None), ("balloon", .5, None),
                                            ("balloon", 1, "cached-run")])
def test_transformed_images_cannot_reuse_untransformed_teacher_logits(task, alpha, cache):
    with pytest.raises(ValueError, match="hard labels only"):
        check_augmentation("balloon-geometric", dict(task=task, alpha=alpha), cache)


def test_default_augmentation_preserves_distillation_path():
    check_augmentation("none", dict(task="goal", alpha=.5), "cache")
    check_augmentation("balloon-geometric", dict(task="balloon", alpha=1), None)
    with pytest.raises(ValueError, match="square RGB"):
        geometric_crop(np.zeros((8, 9, 3)), np.random.default_rng(42))
