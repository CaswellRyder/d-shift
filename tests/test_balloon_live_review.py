import numpy as np
import pytest

from scripts.pi_balloon_live_bench import frame_timing
from scripts.review_balloon_live import sample_health, validate_rows


def fixture():
    rows = [
        dict(
            index=i,
            **frame_timing({"SensorTimestamp": i * 100 + 1}, i * 100 + 2, i * 100 + 3, i * 100 + 4),
        )
        for i in range(2)
    ]
    report = dict(
        machine="armv6l",
        camera_used=True,
        accuracy_measured=False,
        test_evaluated=False,
        deployment_approved=False,
        flight_commands=None,
        camera_rotation_degrees=180,
        queue=False,
        processing_size=[320, 240],
        requested_sensor_size=[2592, 1944],
        frames=2,
        elapsed_s=1,
        observed_live_fps=2,
        invalid_timestamp_frames=0,
        increasing_sensor_timestamps=True,
    )
    return report, rows


def test_darkness_diagnostic_does_not_label_objects_or_certify_lens_obstruction():
    rgb = np.zeros((240, 320, 3), np.uint8)
    assert sample_health(rgb)["nearly_black"]
    rgb[:] = (0, 0, 200)
    assert not sample_health(rgb)["nearly_black"]
    with pytest.raises(ValueError, match="sample"):
        sample_health(rgb.astype(np.float32))


def test_temporal_evidence_recomputed():
    report, rows = fixture()
    assert validate_rows(report, rows)["temporal_observation_valid"]


@pytest.mark.parametrize("fault", ["claim", "age", "valid", "summary", "rate", "order", "size"])
def test_corrupted_evidence_rejected(fault):
    report, rows = fixture()
    if fault == "claim":
        report["accuracy_measured"] = True
    elif fault == "age":
        rows[0]["sensor_to_result_ms"] += 1
    elif fault == "valid":
        rows[0]["timestamp_valid"] = False
    elif fault == "summary":
        report["invalid_timestamp_frames"] = 1
    elif fault == "rate":
        report["observed_live_fps"] = 99
    elif fault == "order":
        rows.reverse()
    elif fault == "size":
        report["processing_size"] = [640, 480]
    with pytest.raises(ValueError):
        validate_rows(report, rows)
