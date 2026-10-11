"""Verify the fixed Astek ablation and reduce development reports, never qualify."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS


def checked_metrics(result):
    total = {label:dict(tp=0,fp=0,fn=0) for label in LABELS}
    for frame in result["frames"]:
        counts = detection_counts(frame["truth"],frame["detections"])
        if counts != frame["counts"]:
            raise ValueError("Frame matching receipt differs from detections")
        for label in LABELS:
            for key in total[label]:
                total[label][key] += counts[label][key]
    calculated = metrics(total)
    if calculated != result["metrics"]:
        raise ValueError("Aggregate metrics differ from frame counts")
    return calculated


def summarize(panels):
    names = {f"{a}{s}" for a in ("control","admit") for s in (47,48)}
    models,manifests = {},{}
    for arm in ("control","admit"):
        manifest = Path(f"data/balloon-astek-{arm}-20261011/manifest.json")
        doc = read_json(manifest)
        added = [r for r in doc["samples"] if "instance_ablation_mode" in r]
        if len(added) != 84 or {r["instance_ablation_mode"] for r in added} != {arm}:
            raise ValueError("Unexpected training addition")
        manifests[arm] = dict(path=str(manifest),sha256=sha256(manifest),
                              counts=dict(Counter(f"{r['split']}/{r['label']}" for r in doc["samples"])))
        for seed in (47,48):
            run = Path(f"runs/balloon-astek-{arm}-seed{seed}-20261011")
            provenance = read_json(run/"provenance.json")
            model = run/"latest.fp32.tflite"
            meta = read_json(model.with_suffix(".json"))
            if (read_json(run/"status.json")["state"] != "complete"
                    or provenance["manifest_sha256"] != sha256(manifest)
                    or provenance["config"]["seed"] != seed or provenance["epochs"] != 8
                    or provenance["learning_rate"] != .0001 or provenance["alpha"] != .5
                    or provenance["hard_label_rows"] != dict(field="instance_ablation_mode",count=84)
                    or meta["sha256"] != sha256(model) or meta["source_checkpoint"] != "latest.keras"
                    or meta["source_student_sha256"] != sha256(run/"latest.keras")
                    or meta["deployment_approved"] is not False):
                raise ValueError("Training or fixed-epoch export identity changed")
            models[f"{arm}{seed}"] = dict(path=str(model),sha256=sha256(model),bytes=model.stat().st_size,
                                          provenance_sha256=sha256(run/"provenance.json"),provenance=provenance)
    if manifests["control"]["counts"] != manifests["admit"]["counts"]:
        raise ValueError("Control class counts differ")
    for key in ("initial_student_sha256","teacher_sha256","training_script_sha256","models_source_sha256"):
        if len({v["provenance"][key] for v in models.values()}) != 1:
            raise ValueError("Training arms differ beyond reviewed data and seed")
    compact = {}
    for path in panels:
        doc = read_json(path)
        if (set(doc["results"]) != names | {"incumbent"}
                or doc["selection"] != "split_merged" or doc["split_max_aspect"] != 2.42913421076193
                or doc["test_evaluated"] is not False or doc["deployment_approved"] is not False):
            raise ValueError("Unexpected comparison scope")
        results = {}
        for name,result in doc["results"].items():
            if result["threshold"] != .8:
                raise ValueError("Threshold changed")
            expected = (models[name]["sha256"] if name in names else
                        "b80460a7526173218b761b265176d19b576d0fa24be14250bf2423e0ad607530")
            if result["model_sha256"] != expected or set(result["full_frame"]) != {"mser_confirmed_parts"}:
                raise ValueError("Model or search identity changed")
            variant = result["full_frame"]["mser_confirmed_parts"]
            if len(variant["frames"]) != doc["reviewed_frame_count"]:
                raise ValueError("Incomplete frame results")
            results[name] = checked_metrics(variant)
        compact[str(path)] = dict(sha256=sha256(path),frames=doc["reviewed_frame_count"],
                                  queue_sha256=doc["queue_sha256"],review_sha256=doc["review_sha256"],results=results)
    return dict(manifests=manifests,models=models,panels=compact,summary_script_sha256=sha256(__file__),
                scope="Fixed final epoch; training-only public addition; repeatedly inspected development panels, not independent qualification",
                independently_verified_sessions=False,test_evaluated=False,pi_tested=False,
                deployment_approved=False,qualification_achieved=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output,summarize(a.panel))
