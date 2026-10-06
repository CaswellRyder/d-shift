"""Verify committed v1 weights and optionally downloaded release assets. No network."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_file(root, name, info):
    root = root.resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path escapes root: {name}")
    data = path.read_bytes()
    if len(data) != info["bytes"] or sha256(data) != info["sha256"]:
        raise ValueError(f"Checksum/size mismatch: {name}")
    return data


def verify(root, assets=None):
    manifest = json.loads((root / "models/v1.0.0/manifest.json").read_text())
    if manifest["deployment_approved"]:
        raise ValueError("This snapshot must remain unapproved for deployment")
    for name, info in manifest["files"].items():
        check_file(root, name, info)
    for name, expected in manifest["source_file_sha256"].items():
        path = root / name
        check_file(root, name, {"bytes": path.stat().st_size, "sha256": expected})
    if assets:
        for name, info in manifest["release_assets"].items():
            check_file(assets, name, info)
            with zipfile.ZipFile(assets / name) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(info["contents"]) | {"CONTENTS.json"}:
                    raise ValueError("Unexpected or duplicate archive contents")
                if json.loads(archive.read("CONTENTS.json")) != info["contents"]:
                    raise ValueError("Internal manifest mismatch")
                for member, expected in info["contents"].items():
                    if Path(member).is_absolute() or ".." in Path(member).parts:
                        raise ValueError("Unsafe archive member")
                    data = archive.read(member)
                    if len(data) != expected["bytes"] or sha256(data) != expected["sha256"]:
                        raise ValueError(f"Archive member mismatch: {member}")
    return {"files_verified": len(manifest["files"]),
            "source_files_verified": len(manifest["source_file_sha256"]),
            "assets_verified": len(manifest["release_assets"]) if assets else 0}


def smoke(root):
    import keras
    import numpy as np
    from dtr.runtime import Predictor, softmax

    # Generated numerical probes, not a dataset or recognition-accuracy test.
    image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    results = {}
    for task, formats in (("goal", ("float", "int8")), ("balloon", ("int8",))):
        directory = root / "models/v1.0.0" / task
        model = keras.models.load_model(directory / "student.keras", compile=False)
        reference = softmax(np.asarray(model(image[None].astype(np.float32), training=False))[0])
        for encoding in formats:
            predictor = Predictor(directory / f"student.{encoding}.tflite", allow_unvalidated=True)
            result = predictor.predict(image)
            scores = np.asarray(result["scores"])
            if not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1):
                raise ValueError("Invalid inference scores")
            delta = float(np.max(np.abs(scores-reference)))
            if encoding == "float" and delta > .001:
                raise ValueError("FP32 export differs from selected Keras checkpoint")
            if result["deployment_approved"]:
                raise ValueError("Unexpected deployment approval")
            results[f"{task}-{encoding}"] = {"parameters": model.count_params(),
                                            "max_score_delta_vs_keras": delta}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run generated probes with local ML dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = verify(root, args.assets)
    if args.smoke:
        result["generated_probe_checks"] = smoke(root)
        result["scope"] = "Packaging/inference verification only; not new accuracy or Pi timing"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
