import numpy as np
import pytest

from scripts.mine_balloon_containment import (
    eligible_images,
    mine,
    pair_choices,
    provisional_targets,
)


def image(name, identifier=1):
    return dict(file_name=name, id=identifier, width=320, height=240)


def test_reserved_family_split_siblings_and_used_groups_excluded_before_pixels():
    docs = dict(
        train=dict(
            images=[
                image("IMG_0001_jpg.rf.aa.jpg"),
                image("frame_0123_jpg.rf.aa.jpg"),
                image("frame_0123_jpg.rf.bb.jpg"),
                image("frame_0456_jpg.rf.aa.jpg"),
                image("frame_0789_jpg.rf.aa.jpg"),
                image("frame_000001_jpg.rf.aa.jpg"),
            ]
        ),
        valid=dict(images=[image("frame_0456_jpg.rf.bb.jpg")]),
        test=dict(images=[image("frame_0789_jpg.rf.cc.jpg")]),
    )
    assert [r["file_name"] for r in eligible_images(docs, {"frame_000001_jpg.jpg"})] == [
        "frame_0123_jpg.rf.aa.jpg"
    ]


def test_only_supported_training_families_are_eligible():
    docs = dict(
        train=dict(
            images=[image("other.png"), image("frame_12345_jpg.jpg"), image("img_000123_jpg.jpg")]
        ),
        valid=dict(images=[]),
        test=dict(images=[]),
    )
    assert eligible_images(docs, set()) == []


def test_upstream_boxes_scaled_without_claiming_review():
    truth = provisional_targets(
        dict(width=640, height=480),
        [dict(id=7, category_id=3, bbox=[20, 40, 60, 80])],
        {3: "blue_ballon"},
    )
    assert truth == [dict(id=7, label="blue_balloon", box=[10, 20, 40, 60])]


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [-1, 0, 2, 2], [0, 0, 321, 240], [0, 0, np.nan, 20]])
def test_invalid_annotation_geometry_rejected(box):
    with pytest.raises(ValueError):
        provisional_targets(
            image("frame_1234_jpg.jpg"), [dict(id=1, category_id=3, bbox=box)], {3: "red_ballon"}
        )


def test_other_classes_not_silently_assigned_to_balloons():
    with pytest.raises(ValueError):
        provisional_targets(
            image("frame_1234_jpg.jpg"),
            [dict(id=1, category_id=3, bbox=[0, 0, 20, 20])],
            {3: "goal"},
        )


def candidate(box):
    return dict(box=box, label="blue_balloon", accepted=True, score=0.99)


def target(box, identifier):
    return dict(box=box, label="blue_balloon", id=identifier)


def test_parent_child_and_keep_both_cases_distinguished():
    parent, child = [0, 0, 100, 100], [20, 20, 40, 40]
    rows = [candidate(parent), candidate(child)]
    assert pair_choices(rows, [target(parent, 0)])[0]["choice"] == "parent"
    assert pair_choices(rows, [target(child, 1)])[0]["choice"] == "child"
    both = pair_choices(rows, [target(parent, 0), target(child, 1)])[0]
    assert both["choice"] == "keep_both" and both["target_ids"] == [0, 1]
    assert "raw_accepted" not in rows[0]


def test_unknown_unaccepted_or_different_colors_not_mined():
    parent, child = [0, 0, 100, 100], [20, 20, 40, 40]
    assert pair_choices([candidate(parent), candidate(child)], []) == []
    assert (
        pair_choices(
            [candidate(parent), dict(candidate(child), accepted=False)], [target(parent, 0)]
        )
        == []
    )
    assert (
        pair_choices(
            [candidate(parent), dict(candidate(child), label="red_balloon")], [target(parent, 0)]
        )
        == []
    )


@pytest.mark.parametrize(
    "sources,review", [(0, 48), (801, 48), (True, 48), (20, 0), (20, 97), (20, True)]
)
def test_invalid_limits_fail_without_data_reads(tmp_path, sources, review):
    with pytest.raises(ValueError, match="limits"):
        mine("missing.zip", "missing.json", {}, tmp_path / "queue", sources, review)
    assert not (tmp_path / "queue").exists()


def test_existing_review_queue_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        mine("missing.zip", "missing.json", {}, tmp_path)
