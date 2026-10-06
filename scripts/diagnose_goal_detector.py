"""Partition cached development errors without inference, relabeling or training changes."""

import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.detector_errors import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = read_json(args.report)
    if report["split"] != "val" or report["metrics"]["test_evaluated"]:
        raise ValueError("Only cached development-validation reports are supported")
    result = diagnose(report["frames"], report["standard"])
    result.update(report_sha256=sha256(args.report), model_sha256=report["model_sha256"],
                  epoch=report["epoch"], detector_input_width=report["detector_input_width"],
                  test_evaluated=False)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
