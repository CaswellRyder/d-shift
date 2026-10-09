import copy
import json

import pytest

from scripts.compare_exact_balloon_search import compare


def fixture(tmp_path):
    summary = dict(pooled={}, runs={})
    for method, fps in (("mser", 2.0), ("mser_fast", 3.0)):
        names = [method + suffix for suffix in ("a", "b")]
        summary["pooled"]["new-views-42:" + method] = dict(
            trials=names,
            processing_fps=fps,
            timing=dict(mean_ms=1000 / fps, p95_ms=1200 / fps),
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
                    "opencv",
                    "unique_frames",
                )
            }
            summary["runs"][name] = dict(report=report)
            (tmp_path / (name + ".frames.jsonl")).write_text(
                json.dumps(
                    dict(
                        round=0, detections=[dict(box=[1, 2, 3, 4], area=3.5, proposal_score=1.23)]
                    )
                )
                + "\n"
            )
    return summary


def test_exact_comparison(tmp_path):
    report = compare(fixture(tmp_path), tmp_path)
    assert report["exact_full_detection_parity"]
    assert report["processing_speedup"] == 1.5
    assert report["unique_frames"] == 1


def test_rejects_changed_proposal_score_even_if_detection_box_is_same(tmp_path):
    summary = fixture(tmp_path)
    path = tmp_path / "mser_fasta.frames.jsonl"
    row = json.loads(path.read_text())
    row["detections"][0]["proposal_score"] += 0.01
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Full detections"):
        compare(summary, tmp_path)


def test_rejects_mismatched_identity_and_missing_repeat(tmp_path):
    summary = fixture(tmp_path)
    changed = copy.deepcopy(summary)
    changed["runs"]["mser_fasta"]["report"]["model_sha256"] = "different"
    with pytest.raises(ValueError, match="identities"):
        compare(changed, tmp_path)
    summary["pooled"]["new-views-42:mser_fast"]["trials"].pop()
    with pytest.raises(ValueError, match="two trials"):
        compare(summary, tmp_path)
