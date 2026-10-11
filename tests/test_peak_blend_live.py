from copy import deepcopy

import pytest

from scripts.pi_peak_blend_live import summarize


def rows():
    return [dict(index=i, localizer_calls=1, neural_calls=1,
        detections=[dict(label='blue_balloon', accepted=True)],
        processing_ms=10., search_ms=4., localizer_ms=3., classifier_ms=2., selection_ms=1.,
        capture_and_convert_ms=5., frame_wall_ms=15., timestamp_valid=True,
        sensor_to_result_ms=20., sensor_timestamp_increasing=True, result_boottime_ns=1000000000+i*200000000)
        for i in range(2)]


def test_summary_recounts_raw_workload_and_camera_latency():
    result = summarize(rows(), .4)
    assert result['live_fps'] == 5 and result['neural_calls'] == 2
    assert result['accepted_counts'] == dict(red_balloon=0, blue_balloon=2)
    assert result['result_intervals_ms']['mean_ms'] == 200
    assert result['sensor_to_result']['mean_ms'] == 20


@pytest.mark.parametrize('field,value', [('index',5), ('localizer_calls',2), ('neural_calls',2), ('processing_ms',11.)])
def test_summary_rejects_workload_or_timing_mismatch(field, value):
    changed = deepcopy(rows())
    changed[1][field] = value
    with pytest.raises(ValueError, match='row'):
        summarize(changed, .4)


def test_invalid_sensor_timestamps_remain_unknown():
    changed = rows()
    for row in changed:
        row.update(timestamp_valid=False, sensor_to_result_ms=None, sensor_timestamp_increasing=False)
    result = summarize(changed, .4)
    assert result['invalid_timestamp_frames'] == 2 and result['sensor_to_result'] is None
    assert not result['increasing_sensor_timestamps']


@pytest.mark.parametrize('elapsed', [0, -1, float('nan')])
def test_elapsed_must_be_positive_and_finite(elapsed):
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(rows(), elapsed)
