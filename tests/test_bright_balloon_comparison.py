import json

import pytest

from scripts.summarize_bright_balloon_search import compare


def fixture(root):
    summary = dict(pooled={}, runs={})
    for method, latency in (("mser_direct", 300.0), ("mser_bright", 200.0)):
        names = [method + suffix for suffix in ("a", "b")]
        summary["pooled"]["new-views-42:" + method] = dict(
            trials=names, processing_fps=1000 / latency, timing=dict(mean_ms=latency)
        )
        for name in names:
            report = {
                k: "same"
                for k in (
                    "model_sha256",
                    "bundle_sha256",
                    "inputs_sha256",
                    "golden_sha256",
                    "runtime_library_sha256",
                    "script_sha256",
                    "region_library_sha256",
                    "opencv",
                    "unique_frames",
                )
            }
            report.update(
                search=method, golden_search="mser" if method == "mser_direct" else method
            )
            summary["runs"][name] = dict(report=report)
            detection = dict(label="background", accepted=False, score=latency / 1000)
            (root / (name + ".frames.jsonl")).write_text(
                json.dumps(
                    dict(
                        round=0,
                        detections=[detection],
                        search_ms=latency - 10,
                        inference_ms=10.0,
                        neural_calls=1,
                    )
                )
                + "\n"
            )
    inputs = dict(frames=[dict(source="fake.png", panel="synthetic", truth=[])])
    return summary, inputs


def test_same_counts_do_not_become_semantic_equivalence(tmp_path):
    summary, inputs = fixture(tmp_path)
    result = compare(summary, tmp_path, inputs)
    assert result["processing_speedup"] == 1.5
    assert result["same_counts_on_each_development_frame"]
    assert result["frames_with_changed_detections"] == 1
    assert not result["general_semantic_equivalence"]
    assert not result["deployment_approved"] and not result["flight_qualified"]


@pytest.mark.parametrize("fault", ["golden", "identity", "missing", "repeat", "library"])
def test_bad_policy_evidence_fails(tmp_path, fault):
    summary, inputs = fixture(tmp_path)
    report = summary["runs"]["mser_brighta"]["report"]
    if fault == "golden":
        report["golden_search"] = "mser"
    elif fault == "identity":
        report["model_sha256"] = "changed"
    elif fault == "missing":
        summary["pooled"]["new-views-42:mser_bright"]["trials"].pop()
    elif fault == "library":
        report["region_library_sha256"] = None
    else:
        path = tmp_path / "mser_brightb.frames.jsonl"
        row = json.loads(path.read_text())
        row["detections"][0]["score"] = 0.0
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        compare(summary, tmp_path, inputs)
