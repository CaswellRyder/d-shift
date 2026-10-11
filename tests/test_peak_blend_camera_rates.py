from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import pi_peak_blend_camera_rates as runner
from scripts.pi_peak_blend_live import summarize


def blocks():
    result = []
    for fps in runner.ORDER:
        rows = [dict(index=i, localizer_calls=1, neural_calls=0, detections=[], processing_ms=10.,
            search_ms=4., localizer_ms=3., classifier_ms=2., selection_ms=1., capture_and_convert_ms=5.,
            frame_wall_ms=15., timestamp_valid=True, sensor_to_result_ms=1000/fps, sensor_timestamp_increasing=True,
            result_boottime_ns=1000000000+i*400000000, exposure_us=1000, frame_duration_us=1000000/fps)
            for i in range(24)]
        result.append(dict(requested_fps=fps, rows=rows, summary=summarize(rows, 120/fps),
            before=dict(health=dict(throttled='throttled=0x0')), after=dict(health=dict(throttled='throttled=0x0'))))
    return result


def test_aggregate_uses_all_blocks_and_elapsed_not_mean_fps():
    data = blocks()
    result = runner.aggregate(data)
    assert result['rates']['5']['frames'] == result['rates']['10']['frames'] == 48
    assert result['rates']['5']['live_fps'] == 1
    assert result['rates']['10']['live_fps'] == 2
    assert result['eligible_for_further_10fps_research']
    data[1]['summary'] = summarize(data[1]['rows'], 60)
    assert not runner.aggregate(data)['eligible_for_further_10fps_research']


def test_age_regression_or_throttling_blocks_research_preference():
    data = blocks()
    data[2]['after']['health']['throttled'] = 'throttled=0x50000'
    assert not runner.aggregate(data)['eligible_for_further_10fps_research']
    data = blocks()
    for row in data[2]['rows']:
        row['sensor_to_result_ms'] = 1000
    data[2]['summary'] = summarize(data[2]['rows'], 12)
    assert not runner.aggregate(data)['eligible_for_further_10fps_research']


def test_wrong_order_or_missing_frames_rejected():
    data = blocks()
    with pytest.raises(ValueError, match='order'):
        runner.aggregate(data[1:]+data[:1])
    data[1]['rows'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        runner.aggregate(data)


def test_capture_is_sequential_retains_rgb_and_closes_camera(monkeypatch):
    calls = []
    transform = SimpleNamespace(hflip=True, vflip=True, transpose=False)

    class Camera:
        def create_video_configuration(self, **config):
            calls.append(('config', config['controls']['FrameRate']))
            self.config = config
            return config

        def configure(self, config):
            assert config is self.config

        def camera_configuration(self):
            return self.config

        def start(self):
            calls.append('start')

        def close(self):
            calls.append('close')

    expected = blocks()[0]
    pixels = [np.zeros((240,320,3), np.uint8) for _ in range(24)]

    def capture(camera, infer, count, seconds):
        assert count == 24 and seconds == 30
        for rgb in pixels:
            infer(rgb)
        return deepcopy(expected['rows']), [], 24

    def infer(rgb):
        calls.append('infer')
        return {}

    monkeypatch.setattr(runner, 'run_camera', capture)
    monkeypatch.setattr(runner.time, 'sleep', lambda n: calls.append(('settle', n)))
    monkeypatch.setattr(runner, 'telemetry', lambda: expected['before'])
    block, retained = runner.camera_block(Camera, transform, infer, 5)
    assert calls == [('config',5), 'start', ('settle',1)]+['infer']*24+['close']
    assert all(a is b for a,b in zip(pixels, retained, strict=True))
    assert block['summary']['frames'] == 24
    calls.clear()

    def broken(*args):
        raise TimeoutError('capture timeout')

    monkeypatch.setattr(runner, 'run_camera', broken)
    with pytest.raises(TimeoutError):
        runner.camera_block(Camera, transform, infer, 10)
    assert calls[-1] == 'close'
