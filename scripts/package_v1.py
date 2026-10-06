"""Freeze selected v1 weights and ancestry, without datasets or pixel source.

Copies immutable existing artifacts; never trains or changes original metadata.
Run once from a checkout with the original research artifacts present.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
TEACHERS = [f"runs/{task}-{stage}" for task in ("goal", "balloon")
            for stage in ("dtr-v10-20261003", "proposals-20261005")]
STUDENTS = {"goal": "runs/student-research-20261005/context-kd",
            "balloon": "runs/balloon-pi-student-20261005"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def check(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed artifact: {path}")


def main():
    target = ROOT / "models" / f"v{VERSION}"
    output = ROOT / "output" / f"release-v{VERSION}"
    if target.exists() or output.exists():
        raise FileExistsError("Version output already exists; do not replace a frozen release")
    weights = {}
    lineage = []

    def include(relative):
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected regular source file: {relative}")
        weights[relative] = path

    backbone = "artifacts/pretrained/mobilenetv4_conv_small"
    check(ROOT / f"{backbone}.keras", read(ROOT / f"{backbone}.json")["sha256"])
    for suffix in ("keras", "json"):
        include(f"{backbone}.{suffix}")

    for run in TEACHERS:
        meta = read(ROOT / run / "teacher.json")
        provenance = read(ROOT / run / "provenance.json")
        check(ROOT / run / "teacher.keras", meta["sha256"])
        parent = Path(provenance["pretrained"])
        check(parent, provenance["pretrained_sha256"])
        parent_relative = str(parent.relative_to(ROOT))
        if parent_relative not in weights:
            raise ValueError("Parent must be included before descendant")
        lineage.append(dict(stage=run, parent=parent_relative,
                            parent_sha256=provenance["pretrained_sha256"],
                            artifact=f"{run}/teacher.keras", sha256=meta["sha256"],
                            data_manifest_sha256=provenance["manifest_sha256"]))
        for name in ("teacher.keras", "teacher.json", "provenance.json", "config.json",
                     "head/epochs.csv", "finetune/epochs.csv"):
            include(f"{run}/{name}")

    for task, run in STUDENTS.items():
        provenance = read(ROOT / run / "provenance.json")
        parent = Path(provenance["teacher"])
        check(parent, provenance["teacher_sha256"])
        if str(parent.relative_to(ROOT)) not in weights:
            raise ValueError("Student teacher is absent from archive")
        names = ["student.keras", "student.int8.tflite", "student.int8.json",
                 "provenance.json", "report.json", "epochs.csv"]
        if task == "goal":
            names += ["student.float.tflite", "student.float.json", "frames-mac.json"]
            check(ROOT / run / "student.keras",
                  read(ROOT / run / "student.float.json")["source_student_sha256"])
        for name in names:
            include(f"{run}/{name}")
            if name.endswith(".tflite"):
                check(ROOT / run / name, read((ROOT / run / name).with_suffix(".json"))["sha256"])
        lineage.append(dict(stage=run, parent=str(parent.relative_to(ROOT)),
                            parent_sha256=provenance["teacher_sha256"],
                            artifact=f"{run}/student.keras", sha256=digest(ROOT / run / "student.keras"),
                            data_manifest_sha256=provenance["manifest_sha256"]))

    # No image files, datasets, train-target caches, raw frame outputs, or pixel code.
    output.mkdir(parents=True)
    target.mkdir(parents=True)
    for task, run in STUDENTS.items():
        (target / task).mkdir()
        for relative, path in weights.items():
            if str(path.parent) == str(ROOT / run) and path.name != "frames-mac.json":
                shutil.copyfile(path, target / task / path.name)
    write(target / "lineage.json", {"version": VERSION, "edges": lineage,
          "scope": "Selected stage checkpoints, not every epoch or rejected experiment; datasets excluded",
          "deployment_approved": False})

    evidence = {
        "goal-full-frame.json": "runs/student-research-20261005/context-kd/frames-mac.json",
        "static-comparison.json": "runs/competition-comparison-20261006/pi-results/static.json",
        "paced-comparison.json": "runs/competition-comparison-20261006/pi-results/paced-v2.json",
        "camera-comparison.json": "runs/competition-comparison-20261006/pi-results/camera/report.json",
        "runtime-build.json": "runs/tflite-rebuild-20261006/build-receipt-final.json",
    }
    (target / "evidence").mkdir()
    for name, source in evidence.items():
        shutil.copyfile(ROOT / source, target / "evidence" / name)
    for source in (ROOT / "runs/tflite-rebuild-20261006").glob("*-crops-*.json"):
        shutil.copyfile(source, target / "evidence" / source.name)
    for source in (ROOT / "runs/tflite-rebuild-20261006").glob("*-replay*.json"):
        shutil.copyfile(source, target / "evidence" / source.name)

    def archive(name, items):
        archive_path = output / name
        hashes = {relative: {"sha256": digest(path), "bytes": path.stat().st_size}
                  for relative, path in sorted(items.items())}
        with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as bundle:
            for relative, path in sorted(items.items()):
                bundle.write(path, relative)
            bundle.writestr("CONTENTS.json", json.dumps(hashes, indent=2) + "\n")
        return {"sha256": digest(archive_path), "bytes": archive_path.stat().st_size,
                "contents": hashes}

    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "docs/MODEL_V1.md"):
        include(name)
    assets = {f"d-shift-v{VERSION}-weight-lineage.zip":
              archive(f"d-shift-v{VERSION}-weight-lineage.zip", weights)}
    runtime_root = ROOT / "runs/tflite-rebuild-20261006"
    check(runtime_root / "libtensorflowlite_c.candidate.so",
          read(ROOT / f"configs/releases/v{VERSION}.json")["runtime_sha256"])
    runtime = {"libtensorflowlite_c.candidate.so": runtime_root / "libtensorflowlite_c.candidate.so",
               "build-receipt-final.json": runtime_root / "build-receipt-final.json"}
    for path in (runtime_root / "licenses").rglob("*"):
        if path.is_file():
            runtime[str(path.relative_to(runtime_root))] = path
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "docs/TFLITE_REBUILD.md"):
        runtime[name] = ROOT / name
    assets[f"d-shift-v{VERSION}-armv6-runtime.zip"] = archive(
        f"d-shift-v{VERSION}-armv6-runtime.zip", runtime)
    files = {str(p.relative_to(ROOT)): {"sha256": digest(p), "bytes": p.stat().st_size}
             for p in sorted(target.rglob("*")) if p.is_file()}
    source_files = [p for folder in ("src", "scripts", "configs")
                    for p in (ROOT / folder).rglob("*")
                    if p.is_file() and p.suffix in (".py", ".json", ".cmake", ".sh", ".patch", ".h", ".cc")]
    write(target / "manifest.json", dict(version=VERSION, deployment_approved=False,
          base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
          source_binding="Final release tag pins packaging/docs; source_file_sha256 pins implementation",
          source_file_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(source_files)},
          files=files, release_assets=assets))
    with (output / "SHA256SUMS").open("x") as stream:
        for name, info in sorted(assets.items()):
            stream.write(f"{info['sha256']}  {name}\n")
    print(json.dumps({"models": str(target), "assets": {k: v['bytes'] for k, v in assets.items()}}, indent=2))


if __name__ == "__main__":
    main()
