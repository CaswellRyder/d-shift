"""Create an exclusive provenance receipt for the isolated runtime build."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", default="build-receipt.json")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "tensorflow"
    scripts = Path(__file__).resolve().parent
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != "72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa":
        raise ValueError("Unexpected TensorFlow revision")
    changes = subprocess.check_output(["git", "-C", str(source), "diff"], text=True)
    dependencies = {}
    for directory in sorted((root / "build-armv6").iterdir()):
        if (directory / ".git").exists():
            dependencies[directory.name] = subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    paths = [root / "build-armv6/libtensorflowlite_c.so", root / "build-armv6/CMakeCache.txt",
             root / "build-armv6/compile_commands.json", root / "host-tools/bin/flatc",
             root / "armv6-smoke", root / "pi-sysroot.tar.gz"]
    paths += sorted(root.glob("build-attempt-*.log"))
    receipt = dict(source_revision=revision, source_patch=changes, dependencies=dependencies,
                   files={str(p.relative_to(root)): digest(p) for p in paths},
                   recipe_sha256={p.name: digest(p) for p in scripts.iterdir() if p.is_file()},
                   target="arm-linux-gnueabihf.2.28 / arm1176jzf_s", zig="0.14.1",
                   cmake="3.31.6", ninja="1.11.1.3", installed_runtime_replaced=False,
                   fast_math=False, deployment_approved=False)
    with (root / args.output).open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(root / args.output)


if __name__ == "__main__":
    main()
