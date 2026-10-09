"""Build research-only MSER reducer locally; never install a system library."""

import argparse
from pathlib import Path
import platform
import subprocess

from dtr.data import sha256, write_json


def build(source, output, compiler="cc"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O3", "-std=c99", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared"]
    if platform.machine() == "armv6l":
        command.extend(["-mcpu=arm1176jzf-s", "-mfpu=vfp", "-mfloat-abi=hard"])
    command.extend([str(source), "-o", str(output)])
    subprocess.run(command, check=True, timeout=60)
    receipt = dict(
        contract="dtr-balloon-regions-v1",
        sha256=sha256(output),
        source_sha256=sha256(source),
        compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        command=command,
        machine=platform.machine(),
        deployment_approved=False,
    )
    write_json(output.with_suffix(".json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build(args.source, args.output), flush=True)
