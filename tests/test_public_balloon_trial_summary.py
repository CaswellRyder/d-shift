from copy import deepcopy

import pytest

from scripts.evaluate_red_blue_development import detection_counts,metrics
from scripts.summarize_public_balloon_trial import checked_metrics


def fixture():
    truth=[dict(label="blue_balloon",box=[0,0,10,10])]
    detections=[dict(label="blue_balloon",box=[0,0,10,10],score=.9,accepted=True),
                dict(label="red_balloon",box=[30,30,40,40],score=.85,accepted=True)]
    counts=detection_counts(truth,detections)
    return dict(metrics=metrics(counts),frames=[dict(truth=truth,detections=detections,counts=counts)])


def test_summary_recomputes_matching_and_metrics():
    result=fixture()
    actual=checked_metrics(result)
    assert actual["red_balloon"]["fp"]==1 and actual["blue_balloon"]["tp"]==1


@pytest.mark.parametrize("level",["frame","aggregate"])
def test_summary_rejects_changed_counts(level):
    result=deepcopy(fixture())
    if level=="frame":
        result["frames"][0]["counts"]["blue_balloon"]["tp"]=0
    else:
        result["metrics"]["blue_balloon"]["recall"]=.5
    with pytest.raises(ValueError):
        checked_metrics(result)
