"""Wrap a frozen replay runner, recording whole-process Linux resource usage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys


def main():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This resource report uses Linux RSS units")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner = args.runner.resolve()
    output = Path(args.arguments[args.arguments.index("--output") + 1]).with_suffix(".resources.json")
    if output.exists():
        raise FileExistsError(output)
    library = Path(os.environ["DTR_TFLITE_LIBRARY"]).resolve()
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), *args.arguments]
    runpy.run_path(str(runner), run_name="__main__")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = dict(library=str(library), library_sha256=library_hash,
                  runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                  peak_rss_kib=usage.ru_maxrss, major_faults=usage.ru_majflt,
                  minor_faults=usage.ru_minflt, user_seconds=usage.ru_utime,
                  system_seconds=usage.ru_stime,
                  scope="Whole Linux process including imports, validation, decoding and replay; not incremental runtime RAM")
    if hashlib.sha256(library.read_bytes()).hexdigest() != library_hash:
        raise ValueError("Runtime changed during measurement")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
