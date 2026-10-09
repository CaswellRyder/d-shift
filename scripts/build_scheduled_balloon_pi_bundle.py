"""Add research scheduler without changing weights, inputs or policy goldens."""

import argparse
from pathlib import Path
import shutil

from dtr.data import read_json, sha256, write_json
from scripts.pi_red_blue_bench import verify_bundle


def prepare(reference, output):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_bundle(reference)
    golden = read_json(reference / "search-golden.json")
    if any(not g.get("mser_bright") or not g.get("mser") for g in golden.values()):
        raise ValueError("Require both policy goldens")
    for name in read_json(reference / "bundle.json")["files"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
    shutil.copyfile("scripts/pi_balloon_schedule_bench.py", output / "pi_balloon_schedule_bench.py")
    shutil.copyfile(
        "src/dtr/research_search_schedule.py", output / "dtr/research_search_schedule.py"
    )
    manifest = dict(
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
        reference_bundle_sha256=sha256(reference / "bundle.json"),
        builder_sha256=sha256(__file__),
        deployment_approved=False,
        test_evaluated=False,
        scope="Periodic search dispatch research, not temporal recall or flight qualification",
    )
    write_json(output / "bundle.json", manifest)
    verify_bundle(output)
    return dict(path=str(output), bundle_sha256=sha256(output / "bundle.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    print(prepare(args.reference, args.output))
