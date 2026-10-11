"""Create an isolated, policy-bound Pi replay bundle; never replace active deployment."""
import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.pi_balloon_search_bench import SELECTIONS, check_predictions, process_frame
from scripts.pi_red_blue_bench import verify_bundle


def prepare(reference, panels, output, library):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError(output)
    receipt = verify_bundle(reference)
    inputs = read_json(reference / "inputs.json")
    name = inputs["default_model"]
    for relative in receipt["files"]:
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / relative, dest)
    for filename in ("pi_balloon_search_bench.py", "research_balloon_search.py"):
        shutil.copyfile(Path("scripts") / filename, output / filename)
    panel_receipts = []
    for index, panel in enumerate(map(Path, panels)):
        doc = read_json(panel / "inputs.json")
        if not doc.get("split", "").startswith(("development", "lockbox_panel")):
            raise ValueError("Require an already-used development panel, never final test")
        for row in doc["frames"]:
            source = (panel / row["path"]).resolve()
            if not source.is_relative_to(panel.resolve()):
                raise ValueError("Frame escapes panel")
            relative = f"selection-panel-{index}/{row['path']}"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inputs["frames"].append(dict(row, path=relative, panel=doc["split"]))
        panel_receipts.append(dict(inputs_sha256=sha256(panel / "inputs.json"), split=doc["split"]))
    predictor = Predictor(output / inputs["models"][name]["path"], allow_unvalidated=True)
    native = NativeBalloonRegions(library, direct=True)
    golden = {name: {"mser" if p == "parts" else "mser_" + p: [] for p in SELECTIONS}}
    for row in inputs["frames"]:
        with Image.open(output / row["path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        for policy in SELECTIONS:
            key = "mser" if policy == "parts" else "mser_" + policy
            result = process_frame(rgb, predictor, "mser", selection=policy)
            fast = process_frame(rgb, predictor, "mser_direct", native, policy)
            check_predictions(fast["detections"], result["detections"])
            golden[name][key].append(dict(source=row["source"], detections=result["detections"]))
    # This replay bundle exposes only the model for which all frames have goldens.
    inputs["models"] = {name: inputs["models"][name]}
    write_json(output / "inputs.json", inputs)
    write_json(output / "search-golden.json", golden)
    write_json(output / "bundle.json", dict(
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
               if p.is_file() and p != output / "bundle.json"},
        reference_bundle_sha256=sha256(reference / "bundle.json"),
        panels=panel_receipts, builder_sha256=sha256(__file__),
        default_model=name, selection_policies=list(SELECTIONS),
        deployment_approved=False, test_evaluated=False,
        scope="Isolated research replay with policy-specific goldens; no live default changes"))
    verify_bundle(output)
    return dict(path=str(output), frames=len(inputs["frames"]), model=name,
                bundle_sha256=sha256(output / "bundle.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "output", "library"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--panel", action="append", default=[])
    args = p.parse_args()
    print(prepare(args.reference, args.panel, args.output, args.library))
