import pytest

from scripts.build_red_blue_bootstrap import split_assignments


def fixture_inputs():
    return (
        [dict(id=0, source="balloon/train/a.jpg"), dict(id=1, source="balloon/train/a.jpg"),
         dict(id=2, source="balloon/train/b.jpg"), dict(id=3, source="balloon/val/c.jpg")],
        dict(background=[0], red_balloon=[1], blue_balloon=[2]),
        dict(validation_source_anchors=[1],
             reserved_upstream_val=dict(background=[], red_balloon=[3], blue_balloon=[])),
        ["background", "red_balloon", "blue_balloon"],
    )


def test_all_views_from_validation_source_are_held_together():
    rows = split_assignments(*fixture_inputs())
    assert rows == [(0, "background", "val"), (1, "red_balloon", "val"),
                    (3, "red_balloon", "test"), (2, "blue_balloon", "train")]


def test_reserved_test_cannot_use_train_photo():
    index, review, plan, classes = fixture_inputs()
    plan["reserved_upstream_val"]["red_balloon"] = [2]
    with pytest.raises(ValueError, match="upstream val"):
        split_assignments(index, review, plan, classes)


def test_no_duplicate_or_ambiguous_color_assignments():
    index, review, plan, classes = fixture_inputs()
    review["blue_balloon"].append(1)
    with pytest.raises(ValueError, match="Duplicate annotation"):
        split_assignments(index, review, plan, classes)


def test_development_pool_cannot_consume_reserved_source():
    index, review, plan, classes = fixture_inputs()
    review["red_balloon"].append(3)
    with pytest.raises(ValueError, match="upstream train"):
        split_assignments(index, review, plan, classes)
