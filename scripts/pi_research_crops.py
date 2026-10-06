"""Self-contained, bounded crop latency/parity test. No camera or actuation."""
import argparse
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=".")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        parser.error("Require 1..10 rounds")
    bundle, output = Path(args.bundle).resolve(), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    receipt = read_json(bundle / "bundle.json")
    for relative, digest in receipt["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or sha256(path) != digest:
            raise ValueError("Bundle file changed or escaped root")
    golden = read_json(bundle / "golden.json")
    if not golden["crops"] or not golden["models"]:
        raise ValueError("Empty comparison bundle")

    def verified_path(relative):
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or relative not in receipt["files"]:
            raise ValueError("Unverified bundle reference")
        return path

    images = []
    for row in golden["crops"]:
        with Image.open(verified_path(row["path"])) as im:
            images.append(np.array(im.convert("RGB")))
    models = {name: Predictor(verified_path(path), True) for name,path in golden["models"].items()}
    for predictor in models.values():
        for rgb in images[:5]:
            predictor.predict(rgb)
    times = {name: [] for name in models}
    parity = {name: dict(max_score_delta=0., labels_match=True) for name in models}
    for repeat in range(args.rounds):
        for index, (row, rgb) in enumerate(zip(golden["crops"], images)):
            names = list(models)
            # Rotate/reverse execution order to reduce fixed-order bias.
            shift = (repeat+index) % len(names)
            names = names[shift:] + names[:shift]
            if (repeat+index) % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                got = models[name].predict(rgb)
                times[name].append((time.perf_counter()-start)*1000)
                reference = row["reference"][name]
                delta = float(np.max(np.abs(np.array(got["scores"])-reference["scores"])))
                parity[name]["max_score_delta"] = max(parity[name]["max_score_delta"], delta)
                parity[name]["labels_match"] &= got["label"] == reference["label"]
    results = {}
    for name, predictor in models.items():
        tolerance = .03 if predictor.input["dtype"] == np.int8 else .001
        cell = parity[name]
        results[name] = dict(**cell, tolerance=tolerance,
                            parity_passed=cell["labels_match"] and cell["max_score_delta"] <= tolerance,
                            samples=len(times[name]), mean_ms=float(np.mean(times[name])),
                            p50_ms=float(np.median(times[name])),
                            p95_ms=float(np.percentile(times[name],95)),
                            runtime=type(predictor.interpreter).__module__,
                            model_sha256=predictor.metadata["sha256"])
    write_json(output, dict(results=results, host=platform.node(), machine=platform.machine(),
                           bundle_sha256=sha256(bundle / "bundle.json"),
                           scope="Crop resize + inference + softmax; predecoded crops, 5 warmups; not full frame/camera timing",
                           actual_armv6=platform.machine() == "armv6l", flight_commands=None,
                           deployment_approved=False))
    print(results, flush=True)
    if not all(row["parity_passed"] for row in results.values()):
        raise SystemExit("Output parity failed; timings are not a validated comparison")


if __name__ == "__main__":
    main()
