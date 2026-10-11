import pytest

from scripts.audit_peak_blend_camera_rates import cpu_activity, validate_clock_and_rate
from scripts.pi_balloon_live_bench import frame_timing


def row():
    metadata = dict(SensorTimestamp=1000000000, ExposureTime=12000, AnalogueGain=1., FrameDuration=100000)
    return frame_timing(metadata, 990000000, 1100000000, 1300000000)


def test_applied_sensor_rate_and_boottime_are_checked():
    validate_clock_and_rate([row()], 10)
    with pytest.raises(ValueError, match='cadence'):
        validate_clock_and_rate([row()], 5)


@pytest.mark.parametrize('field,value', [
    ('sensor_timestamp_ns',1400000000), ('sensor_to_result_ms',1.),
    ('frame_duration_us', float('nan')), ('frame_duration_us', None),
])
def test_forged_timestamp_or_bad_cadence_rejected(field, value):
    changed = row()
    changed[field] = value
    with pytest.raises(ValueError):
        validate_clock_and_rate([changed], 10)


def test_duplicate_sensor_frame_rejected():
    with pytest.raises(ValueError, match='timestamp'):
        validate_clock_and_rate([row(), row()], 10)


def test_cpu_busy_does_not_double_count_guest_or_iowait():
    result = cpu_activity('cpu 10 0 10 50 5 0 0 0 5 0', 'cpu 30 0 20 100 15 0 10 0 10 0')
    assert result['total_ticks'] == 100 and result['idle_and_iowait_ticks'] == 60
    assert result['busy_percent'] == 40


def test_invalid_cpu_counters_rejected():
    with pytest.raises(ValueError):
        cpu_activity('cpu 0 0', 'cpu 1 2')
    with pytest.raises(ValueError):
        cpu_activity('cpu 10 0 0 0 0 0 0 0', 'cpu 0 0 0 0 0 0 0 0')
