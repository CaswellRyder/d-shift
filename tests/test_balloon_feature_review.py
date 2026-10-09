import copy

import pytest

from scripts.distill_balloon_features import MANIFEST_SHA, TEACHER_SHA
from scripts.review_balloon_feature_experiment import check_pair


def pair():
    control = dict(
        feature_weight=0,
        feature_projection_exported=False,
        initial_weights_sha256="initial",
        teacher_cache={"targets_sha256": "cache"},
        seed=42,
        config={"classes": ["background", "red_balloon", "blue_balloon"]},
        epochs=25,
        learning_rate=0.001,
        shuffle="paired",
        augmentation="none",
        manifest_sha256=MANIFEST_SHA,
        teacher_sha256=TEACHER_SHA,
        training_script_sha256="training",
        models_source_sha256="model",
        input_guard_sha256="guard",
        train_examples=584,
        validation_examples=8,
        selection="minimum validation hard CE",
    )
    return control, dict(copy.deepcopy(control), feature_weight=1)


def test_matched_pair_is_accepted():
    check_pair(*pair())


@pytest.mark.parametrize(
    "key",
    [
        "initial_weights_sha256",
        "teacher_cache",
        "seed",
        "config",
        "epochs",
        "learning_rate",
        "shuffle",
        "augmentation",
        "manifest_sha256",
        "teacher_sha256",
        "training_script_sha256",
        "models_source_sha256",
        "input_guard_sha256",
        "train_examples",
        "validation_examples",
        "selection",
    ],
)
def test_unmatched_pair_is_rejected(key):
    control, hint = pair()
    hint[key] = "changed"
    with pytest.raises(ValueError, match="Unmatched"):
        check_pair(control, hint)


def test_wrong_arms_and_exported_projection_are_rejected():
    control, hint = pair()
    with pytest.raises(ValueError, match="arms"):
        check_pair(hint, control)
    hint["feature_projection_exported"] = True
    with pytest.raises(ValueError, match="projection"):
        check_pair(control, hint)
    control, hint = pair()
    control["manifest_sha256"] = hint["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="Frozen"):
        check_pair(control, hint)
