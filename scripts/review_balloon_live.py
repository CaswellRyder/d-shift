"""Verify live timing artifacts and flag unusable sample imagery, not accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json


def sample_health(rgb):
    if rgb.dtype != np.uint8 or rgb.shape != (240, 320, 3):
        raise ValueError("Expected 320x240 RGB uint8 sample")
    peak = rgb.max(axis=2)
    dark_fraction = float(np.mean(peak < 12))
    return dict(
        mean_rgb=float(rgb.mean()),
        max_channel=int(rgb.max()),
        peak_p95=float(np.percentile(peak, 95)),
        dark_pixel_fraction=dark_fraction,
        nearly_black=dark_fraction >= 0.99,
        scope="Brightness diagnostic only; not lens obstruction diagnosis or target labels",
    )


def validate_rows(report, rows):
    if (
        report.get("machine") != "armv6l"
        or report.get("camera_used") is not True
        or report.get("accuracy_measured") is not False
        or report.get("test_evaluated") is not False
        or report.get("deployment_approved") is not False
        or report.get("flight_commands") is not None
        or report.get("camera_rotation_degrees") != 180
        or report.get("queue") is not False
        or report.get("processing_size") != [320, 240]
        or report.get("requested_sensor_size") != [2592, 1944]
        or report.get("frames") != len(rows)
        or len(rows) < 2
    ):
        raise ValueError("Require bounded ARMv6 live timing, without qualification claims")
    elapsed = report["elapsed_s"]
    if (
        not np.isfinite(elapsed)
        or elapsed <= 0
        or not np.isclose(report["observed_live_fps"], len(rows) / elapsed)
    ):
        raise ValueError("Invalid live elapsed time or rate")
    previous = None
    valid_count = 0
    increasing = True
    for index, row in enumerate(rows):
        if row["index"] != index:
            raise ValueError("Missing or reordered live frames")
        start, acquired, end = (
            row[k]
            for k in (
                "capture_started_boottime_ns",
                "capture_completed_boottime_ns",
                "result_boottime_ns",
            )
        )
        if not 0 < start <= acquired <= end:
            raise ValueError("Invalid boottime ordering")
        sensor = row["sensor_timestamp_ns"]
        valid = type(sensor) is int and 0 < sensor <= acquired
        monotone = valid and (previous is None or sensor > previous)
        if row["timestamp_valid"] != valid or row["sensor_timestamp_increasing"] != monotone:
            raise ValueError("Timestamp evidence does not match reported validity")
        if valid:
            for key, value in (
                ("sensor_to_acquire_ms", (acquired - sensor) / 1e6),
                ("sensor_to_result_ms", (end - sensor) / 1e6),
            ):
                if row[key] != value:
                    raise ValueError("Sensor age does not match timestamp evidence")
            previous = sensor
            valid_count += 1
        elif row["sensor_to_result_ms"] is not None or row["sensor_to_acquire_ms"] is not None:
            raise ValueError("Invalid sensor timestamp has a claimed age")
        increasing = increasing and monotone
    if (
        report["invalid_timestamp_frames"] != len(rows) - valid_count
        or report["increasing_sensor_timestamps"] != increasing
    ):
        raise ValueError("Timestamp summary mismatch")
    return dict(
        valid_timestamp_frames=valid_count,
        strictly_increasing=increasing,
        temporal_observation_valid=valid_count == len(rows) and increasing,
    )


def review(root):
    root = Path(root).resolve()
    report = read_json(root / "report.json")
    for relative, digest in report["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Live artifact hash changed or path escaped")
    if "frames.jsonl" not in report["files"]:
        raise ValueError("Frame log is not hash-bound")
    rows = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    timing_validation = validate_rows(report, rows)
    indices = report.get("saved_sample_indices", [0, len(rows) - 1])
    if (
        not 2 <= len(indices) <= 10
        or len(set(indices)) != len(indices)
        or not {0, len(rows) - 1}.issubset(indices)
        or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
    ):
        raise ValueError("Invalid bounded sample indices")
    names = [f"{i:04d}-raw.png" for i in indices]
    if any(name not in report["files"] for name in names):
        raise ValueError("First/last sample not hash-bound")
    samples = {
        name: sample_health(np.asarray(Image.open(root / name).convert("RGB"))) for name in names
    }
    return dict(
        report=report,
        report_sha256=sha256(root / "report.json"),
        review_script_sha256=sha256(__file__),
        timing_validation=timing_validation,
        samples=samples,
        nearly_black_samples=sum(s["nearly_black"] for s in samples.values()),
        content_scope="Only listed samples retained; not labels for unsaved frames",
        visual_orientation_confirmed=False,
        accuracy_measured=False,
        test_evaluated=False,
        deployment_approved=False,
        flight_qualified=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = review(args.input)
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k not in ("report", "samples")})
