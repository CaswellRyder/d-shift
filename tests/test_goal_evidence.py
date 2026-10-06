import pytest

from dtr.goal_evidence import CLASSES, goal_candidate, goal_evidence


def observation(scores):
    best = max(range(7),key=lambda i:scores[i])
    return dict(label=CLASSES[best],accepted=best != 0 and scores[best] >= .8,
                goal_evidence=goal_evidence(scores,CLASSES))


def test_shape_survives_color_ambiguity_without_relabeling():
    row = observation([.05,.475,0,0,.475,0,0])
    assert not row["accepted"]
    assert row["goal_evidence"]["shape"] == "circle"
    assert row["goal_evidence"]["color"] == "unknown"
    assert row["goal_evidence"]["shape_score"] == pytest.approx(.95)
    assert not goal_candidate(row,"orange",measurement_age_ms=0)["investigation_candidate"]
    for color in ("orange","yellow"):
        result = goal_candidate(row,color,measurement_age_ms=0,allow_unknown_color=True)
        assert result["investigation_candidate"] and not result["confirmed_target"]
        assert not result["action_authorized"] and result["flight_commands"] is None


def test_not_yellow_is_not_orange():
    row = observation([.94,.02,.02,.02,0,0,0])
    result = goal_candidate(row,"orange",measurement_age_ms=0,allow_unknown_color=True)
    assert result["status"] == "insufficient_shape_evidence"
    assert not result["investigation_candidate"]


def test_wrong_color_is_rejected():
    row = observation([.01,.01,0,0,.98,0,0])
    assert goal_candidate(row,"orange",measurement_age_ms=0,allow_unknown_color=True)["status"] == "other_color"
    assert goal_candidate(row,"yellow",measurement_age_ms=0)["confirmed_target"]


def test_color_does_not_cross_shapes():
    row = observation([.1,0,0,.45,0,.45,0])
    assert row["goal_evidence"]["shape"] == "unknown"
    assert row["goal_evidence"]["color"] == "unknown"


@pytest.mark.parametrize("age",[None,-1,501,float("nan"),float("inf")])
def test_stale_or_unverified_measurements_are_ineligible(age):
    row = observation([.05,.475,0,0,.475,0,0])
    assert goal_candidate(row,"orange",measurement_age_ms=age,allow_unknown_color=True)["status"] == "invalid_or_stale"


@pytest.mark.parametrize("extra",[dict(suppressed=True),dict(tracking_valid=False),dict(classification_age_ms=1001)])
def test_suppressed_lost_or_expired_class_is_ineligible(extra):
    row = observation([.05,.475,0,0,.475,0,0])
    assert not goal_candidate(dict(row,**extra),"orange",measurement_age_ms=0,allow_unknown_color=True)["investigation_candidate"]


def test_color_evidence_does_not_override_joint_acceptance():
    row = observation([.14,.76,0,0,.1,0,0])
    assert row["goal_evidence"]["color"] == "orange"
    assert not goal_candidate(row,"orange",measurement_age_ms=0)["confirmed_target"]


def test_taxonomy_order_independent():
    scores = [.05,.475,0,0,.475,0,0]
    assert goal_evidence(scores,CLASSES) == goal_evidence(scores[::-1],CLASSES[::-1])


@pytest.mark.parametrize("scores",[[1]*7,[float("nan")]*7,[-1,2,0,0,0,0,0],[1]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        goal_evidence(scores,CLASSES)
