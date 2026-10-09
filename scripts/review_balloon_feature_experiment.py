"""Bind paired feature-distillation evidence; never promote a model."""

import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.distill_balloon_features import MANIFEST_SHA, TEACHER_SHA, training_boundary


def check_pair(control, hint):
    if control["feature_weight"] != 0 or hint["feature_weight"] != 1:
        raise ValueError("Require control and hint arms")
    for key in (
        "initial_weights_sha256",
        "teacher_cache",
        "seed",
        "config",
        "epochs",
        "learning_rate",
        "shuffle",
        "augmentation",
        "manifest_sha256",
        "teacher_sha256",
        "training_script_sha256",
        "models_source_sha256",
        "input_guard_sha256",
        "train_examples",
        "validation_examples",
        "selection",
    ):
        if control[key] != hint[key]:
            raise ValueError(f"Unmatched paired input: {key}")
    if control["manifest_sha256"] != MANIFEST_SHA or control["teacher_sha256"] != TEACHER_SHA:
        raise ValueError("Frozen input identity changed")
    if any(p["feature_projection_exported"] for p in (control, hint)):
        raise ValueError("Training projection must not be exported")


def graph_contract(path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or inputs[0]["shape"].tolist() != [1, 64, 64, 3]
        or outputs[0]["shape"].tolist() != [1, 3]
    ):
        raise ValueError("Unexpected exported student interface")
    return dict(
        tensors=[
            dict(shape=t["shape"].tolist(), dtype=t["dtype"].__name__)
            for t in interpreter.get_tensor_details()
        ],
        operators=[
            dict(name=o["op_name"], inputs=o["inputs"].tolist(), outputs=o["outputs"].tolist())
            for o in interpreter._get_ops_details()
        ],
    )


def review(root):
    root = Path(root)
    manifest = Path("data/balloon-red-blue-expanded-views-20261008/manifest.json")
    if sha256(manifest) != MANIFEST_SHA:
        raise ValueError("Training manifest changed")
    row_paths = [r["path"] for r in training_boundary(read_json(manifest))]
    panels = {
        p: read_json(root / f"feature-distillation-{p}-scenes.json") for p in ("indoor", "original")
    }
    result, graph_refs, pairs = {}, {}, {}
    for seed in (42, 43):
        provenance = {}
        for arm in ("control", "hint"):
            run = Path(f"runs/balloon-feature-{arm}-seed{seed}-20261008")
            p, report = read_json(run / "provenance.json"), read_json(run / "report.json")
            provenance[arm] = p
            if (
                read_json(run / "status.json")["state"] != "complete"
                or p["teacher_cache"]["row_paths"] != row_paths
                or p["teacher_cache"]["rows"] != len(row_paths)
                or p["seed"] != seed
                or report["student_parameters"] != 7763
                or report["source_student_sha256"] != sha256(run / "student.keras")
            ):
                raise ValueError("Incomplete training or mismatched rows/checkpoint")
            exports = {}
            for dtype in ("fp32", "int8"):
                path = run / f"student.{dtype}.tflite"
                meta = read_json(path.with_suffix(".json"))
                digest = sha256(path)
                name = f"{arm}{seed}-{dtype}"
                if meta["sha256"] != digest or meta["deployment_approved"]:
                    raise ValueError("Export identity or authority mismatch")
                graph = graph_contract(path)
                graph_refs.setdefault(dtype, graph)
                if graph != graph_refs[dtype]:
                    raise ValueError("Exported topology differs between experiment arms")
                scores = {}
                for panel, data in panels.items():
                    scored = data["results"][name]
                    if (
                        scored["model_sha256"] != digest
                        or scored["metadata_sha256"] != sha256(path.with_suffix(".json"))
                        or scored["threshold"] != 0.8
                        or data["test_evaluated"]
                        or data["deployment_approved"]
                    ):
                        raise ValueError("Scene evaluation identity mismatch")
                    scores[panel] = scored["full_frame"]["mser_confirmed_parts"]["metrics"]
                exports[dtype] = dict(sha256=digest, bytes=path.stat().st_size, panels=scores)
            result[f"{arm}{seed}"] = dict(
                run=str(run),
                report=report,
                provenance=p,
                exports=exports,
                files={
                    name: sha256(run / name)
                    for name in (
                        "report.json",
                        "provenance.json",
                        "history.json",
                        "status.json",
                        "student.keras",
                        "student.fp32.tflite",
                        "student.fp32.json",
                        "student.int8.tflite",
                        "student.int8.json",
                    )
                },
            )
        check_pair(provenance["control"], provenance["hint"])
        pairs[str(seed)] = dict(
            matched=True, initial_weights_sha256=provenance["control"]["initial_weights_sha256"]
        )
    return dict(
        scope="Paired development experiment, not independent accuracy or flight qualification",
        pairs=pairs,
        results=result,
        graph_contracts=graph_refs,
        identical_export_topology_within_dtype=True,
        panel_reports={p: sha256(root / f"feature-distillation-{p}-scenes.json") for p in panels},
        reviewer_sha256=sha256(__file__),
        test_evaluated=False,
        deployment_approved=False,
        pi_timing_measured=False,
        qualification_established=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = review(args.root)
    write_json(args.output, report)
    for name, result in report["results"].items():
        print(name, result["exports"]["fp32"]["panels"])
