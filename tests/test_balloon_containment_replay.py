from copy import deepcopy

import pytest

from scripts.compare_balloon_containment import before_parts, compare, confidence_ordered
from scripts.evaluate_red_blue_development import detection_counts, metrics
from scripts.research_balloon_search import suppress_confirmed_balloon_parts


def observations(parent_score=.9, child_score=.99):
    return [dict(box=[0, 0, 100, 100], label="red_balloon", accepted=True, score=parent_score),
            dict(box=[10, 10, 30, 30], label="red_balloon", accepted=True, score=child_score)]


@pytest.mark.parametrize("parent,child,expected", [(.9, .99, [False, True]),
    (.99, .9, [True, False]), (1., 1., [True, False])])
def test_stronger_wins_ties_keep_parent_without_mutating(parent, child, expected):
    raw = observations(parent, child)
    saved = suppress_confirmed_balloon_parts(raw)
    original = deepcopy(saved)
    assert [r["accepted"] for r in confidence_ordered(saved)] == expected
    assert saved == original
    assert all(r["accepted"] for r in before_parts(saved))


@pytest.mark.parametrize("change", [dict(label="blue_balloon"), dict(box=[90, 90, 120, 120]),
                                  dict(box=[10, 10, 90, 90])])
def test_other_colors_neighbors_and_similar_sizes_stay(change):
    rows = observations()
    rows[1].update(change)
    assert all(r["accepted"] for r in confidence_ordered(rows))


def test_ordinary_nms_and_rejected_background_are_never_resurrected():
    rows = observations()
    rows[1].update(accepted=False, suppressed=True)
    rows.append(dict(box=[5, 5, 8, 8], label="background", accepted=False, score=.99))
    assert [r["accepted"] for r in confidence_ordered(rows)] == [True, False, False]


def report():
    truth = [dict(box=[10, 10, 30, 30], label="red_balloon")]
    detections = suppress_confirmed_balloon_parts(observations())
    counts = detection_counts(truth, detections)
    return dict(test_evaluated=False, deployment_approved=False,
        results=dict(candidate=dict(threshold=.8, model_sha256="fixture",
            full_frame=dict(mser_confirmed_parts=dict(metrics=metrics(counts),
                frames=[dict(truth=truth, detections=detections, counts=counts)])))))


def test_replay_checks_original_metrics_and_recovers_stronger_nested_target():
    source = report()
    result = compare(source)["results"]["candidate"]["full_frame"]
    assert result["parent_first"]["metrics"]["red_balloon"]["fn"] == 1
    assert result["confidence_ordered"]["metrics"]["red_balloon"] == dict(
        tp=1, fp=0, fn=0, precision=1., recall=1.)
    assert source == report()


@pytest.mark.parametrize("fault", ["test", "deployment", "threshold", "metrics", "invalid_box", "low_score"])
def test_incompatible_or_corrupt_evidence_fails(fault):
    source = report()
    model = source["results"]["candidate"]
    evidence = model["full_frame"]["mser_confirmed_parts"]
    if fault == "test":
        source["test_evaluated"] = True
    elif fault == "deployment":
        source["deployment_approved"] = True
    elif fault == "threshold":
        model["threshold"] = .7
    elif fault == "metrics":
        evidence["metrics"]["red_balloon"]["tp"] = 99
    elif fault == "invalid_box":
        evidence["frames"][0]["detections"][0]["box"] = [0, 0, 321, 5]
    else:
        evidence["frames"][0]["detections"][1]["score"] = .5
    with pytest.raises(ValueError):
        compare(source)
