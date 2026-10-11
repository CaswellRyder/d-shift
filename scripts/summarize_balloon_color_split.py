"""Verify Lab proposal trials against reviewed development frames; no promotion."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.summarize_public_balloon_trial import checked_metrics


def verify_panel(path, queue, review, model):
    doc = read_json(path)
    frames, decision = reviewed_frames(queue, review)
    references = {r["id"]: r for r, _ in frames}
    if (doc["model_sha256"] != sha256(model)
            or doc["queue_sha256"] != decision["queue_sha256"]
            or doc["review_sha256"] != sha256(review)
            or doc["threshold"] != .8 or doc["matching_iou"] != .5
            or doc["selection"] != "split_merged"
            or doc["split_max_aspect"] != 2.42913421076193
            or doc["test_evaluated"] is not False
            or doc["deployment_approved"] is not False
            or doc["pi_timing_measured"] is not False):
        raise ValueError("Frozen model, review, operating point or research scope changed")
    if set(doc["variants"]) != {"baseline", "color_split", "fragment_guard"}:
        raise ValueError("Missing matched proposal variant")
    results = {}
    for name, result in doc["variants"].items():
        if (len(result["frames"]) != len(references)
                or {f["id"] for f in result["frames"]} != set(references)):
            raise ValueError("Incomplete frame coverage")
        for frame in result["frames"]:
            ref = references[frame["id"]]
            if (frame["truth"] != ref["provisional_truth"]
                    or frame["frame_sha256"] != ref["sha256"]
                    or frame["source"] != ref["source"]):
                raise ValueError("Frame source or reviewed truth changed")
        results[name] = checked_metrics(result)
    return dict(path=str(path), sha256=sha256(path), frames=len(references),
                queue_sha256=doc["queue_sha256"], review_sha256=doc["review_sha256"],
                model_sha256=doc["model_sha256"], search_sha256=doc["script_sha256"],
                baseline_search_sha256=doc["baseline_search_sha256"], results=results)


def summarize(panels, model, training, crop_queue, crop_review, manifest):
    from scripts.prepare_balloon_instance_review import checked_rows
    reviewed = checked_rows(manifest, crop_queue, crop_review)
    decision = read_json(crop_review)
    train = read_json(training)
    if (train["test_evaluated"] is not False or train["deployment_approved"] is not False
            or train["script_sha256"] != sha256("scripts/research_balloon_color_split.py")):
        raise ValueError("Training audit scope or search changed")
    compact = [verify_panel(*panel, model) for panel in panels]
    if any(p["search_sha256"] != train["script_sha256"] for p in compact):
        raise ValueError("Training and development used different search code")
    return dict(
        model_path=str(model), model_sha256=sha256(model),
        training_audit=dict(path=str(training), sha256=sha256(training),
                            coverage={k:v["coverage"] for k,v in train["results"].items()}),
        crop_review=dict(queue_sha256=sha256(Path(crop_queue)/"review.json"),
                         review_sha256=sha256(crop_review), reviewed_background_ids=[r["id"] for r in reviewed],
                         excluded_ids=decision["exclude"], labels_automatically_inferred=False,
                         training_dataset_built=False, retraining_performed=False),
        panels=compact, summary_script_sha256=sha256(__file__),
        decision="Reject Lab search and fragment guard; keep incumbent model and runtime unchanged",
        reason="Extra target coverage is outweighed by scene false positives on every inspected panel",
        followup="Do not label balloon interiors as background. New proposal work must preserve full-instance geometry, not merely create additional colored regions.",
        scope="Repeatedly inspected development panels; overlapping sources, not independent qualification",
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", nargs=3, action="append", required=True, metavar=("REPORT","QUEUE","REVIEW"))
    for name in ("model", "training", "crop-queue", "crop-review", "manifest", "output"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    write_json(a.output, summarize(a.panel,a.model,a.training,a.crop_queue,a.crop_review,a.manifest))
