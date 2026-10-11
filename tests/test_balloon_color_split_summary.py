import copy

import pytest

from scripts import summarize_balloon_color_split as module


@pytest.fixture
def report(monkeypatch):
    frame = dict(id=0, source="development.jpg", frame_sha256="pixels", truth=[], detections=[],
                 counts={k:dict(tp=0,fp=0,fn=0) for k in ("red_balloon","blue_balloon")})
    from scripts.evaluate_red_blue_development import metrics
    result = dict(frames=[frame], metrics=metrics(frame["counts"]))
    doc = dict(model_sha256="model", queue_sha256="queue", review_sha256="review",
               threshold=.8, matching_iou=.5, selection="split_merged", split_max_aspect=2.42913421076193,
               test_evaluated=False, deployment_approved=False, pi_timing_measured=False,
               script_sha256="search", baseline_search_sha256="base",
               variants={k:copy.deepcopy(result) for k in ("baseline","color_split","fragment_guard")})
    monkeypatch.setattr(module,"read_json",lambda path:doc)
    monkeypatch.setattr(module,"sha256",str)
    ref=dict(id=0,source="development.jpg",sha256="pixels",provisional_truth=[])
    monkeypatch.setattr(module,"reviewed_frames",lambda *a:([(ref,None)],dict(queue_sha256="queue")))
    return doc


def test_verified_matched_panels(report):
    out=module.verify_panel("report","queue","review","model")
    assert out["frames"]==1 and len(out["results"])==3


@pytest.mark.parametrize("fault",["model","threshold","missing_variant","missing_frame","truth","counts","metrics"])
def test_rejects_changed_or_incomplete_evidence(report,fault):
    if fault=="model":
        report["model_sha256"]="other"
    elif fault=="threshold":
        report["threshold"] = .7
    elif fault=="missing_variant":
        report["variants"].pop("fragment_guard")
    else:
        result=report["variants"]["color_split"]
        if fault=="missing_frame":
            result["frames"]=[]
        elif fault=="truth":
            result["frames"][0]["truth"]=[dict(label="blue_balloon",box=[0,0,20,20])]
        elif fault=="counts":
            result["frames"][0]["counts"]["blue_balloon"]["fp"]=1
        else:
            result["metrics"]["blue_balloon"]["precision"]=.94
    with pytest.raises(ValueError):
        module.verify_panel("report","queue","review","model")
