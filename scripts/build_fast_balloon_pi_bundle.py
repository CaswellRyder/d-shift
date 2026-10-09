"""Copy a verified research bundle and add exact-search experiment tools."""

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
    old = read_json(reference / "bundle.json")
    for name in old["files"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
    for name in (
        "pi_balloon_search_bench",
        "research_balloon_search_fast",
        "build_native_balloon_regions",
    ):
        shutil.copyfile(f"scripts/{name}.py", output / f"{name}.py")
    for suffix in ("c", "py"):
        name = f"native_balloon_regions.{suffix}"
        shutil.copyfile(Path("src/dtr") / name, output / "dtr" / name)
    manifest = dict(
        files={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
        reference_bundle_sha256=sha256(reference / "bundle.json"),
        builder_sha256=sha256(__file__),
        deployment_approved=False,
        test_evaluated=False,
        scope="Exact-search research; original model, inputs and golden predictions unchanged",
    )
    write_json(output / "bundle.json", manifest)
    return dict(path=str(output), bundle_sha256=sha256(output / "bundle.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.reference, args.output))
