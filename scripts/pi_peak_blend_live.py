"""Bounded full-candidate camera smoke test; every processed RGB frame retained."""
import argparse
import os
from pathlib import Path
import platform
import resource
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.pi_balloon_live_bench import run_camera, validate_camera_config
from scripts.pi_blue_peak_blend import Network, TIMINGS, process
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_red_blue_bench import health, statistics, verify_bundle


def summarize(rows, elapsed):
    if len(rows) < 2 or not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Incomplete live trial')
    for index, row in enumerate(rows):
        if (row['index'] != index or row['localizer_calls'] != 1
                or row['neural_calls'] != len(row['detections'])
                or not 0 <= row['neural_calls'] <= 12
                or not np.isclose(row['processing_ms'], sum(row[k] for k in TIMINGS[1:]), atol=1e-5, rtol=0)):
            raise ValueError('Live row workload or timing invalid')
    valid = [r['sensor_to_result_ms'] for r in rows if r['timestamp_valid']]
    return dict(frames=len(rows), elapsed_s=elapsed, live_fps=len(rows)/elapsed,
        timing={k:statistics([r[k] for r in rows]) for k in (*TIMINGS, 'capture_and_convert_ms', 'frame_wall_ms')},
        sensor_to_result=statistics(valid) if valid else None,
        invalid_timestamp_frames=sum(not r['timestamp_valid'] for r in rows),
        increasing_sensor_timestamps=all(r['sensor_timestamp_increasing'] for r in rows),
        result_intervals_ms=statistics([(b['result_boottime_ns']-a['result_boottime_ns'])/1e6
            for a,b in zip(rows, rows[1:])]),
        accepted_counts={label:sum(d['accepted'] and d['label'] == label for r in rows for d in r['detections'])
            for label in ('red_balloon', 'blue_balloon')},
        neural_calls=sum(r['neural_calls'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'library', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    bundle = verify_bundle(root)
    if bundle['protocol'] != dict(policy='head_blend', count=24, seconds=30, fps=5):
        raise ValueError('Live protocol changed')
    for key in ('DTR_RED_BLUE_COLOR_LOOKUP', 'DTR_RED_BLUE_LOOKUP_LIBRARY'):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    native = NativeBalloonRegions(args.library, direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(root/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source mismatch')
    classifier = Predictor(root/'models/classifier.tflite', allow_unvalidated=True)
    network = Network(root/'models/head_localizer.tflite', True)
    if classifier.metadata['threshold'] != .8:
        raise ValueError('Classifier threshold changed')

    def infer(rgb):
        return process(rgb, classifier, network, native, 'head_blend')

    frames = read_json(root/'decoder-frames.json')
    deltas = []
    for frame in frames[:3]:
        rgb = np.asarray(Image.open(root/frame['path']).convert('RGB'))
        deltas.append(compare(infer(rgb)['detections'], frame['expected']['head_blend']))
    if len(frames) != 141 or len(deltas) != 3:
        raise ValueError('Incomplete pre-camera parity')
    from libcamera import Transform
    from picamera2 import Picamera2
    retained = []

    def collect(rgb):
        result = infer(rgb)
        # capture_rgb returns owned RGB; retain its array without another image copy.
        retained.append(rgb)
        return result

    camera, before = Picamera2(), health()
    try:
        camera.configure(camera.create_video_configuration(
            main=dict(size=(320,240), format='RGB888'),
            sensor=dict(output_size=(2592,1944), bit_depth=10),
            controls=dict(FrameRate=5), buffer_count=2, queue=False,
            transform=Transform(hflip=True, vflip=True)))
        config = camera.camera_configuration()
        validate_camera_config(config)
        camera.start()
        time.sleep(1)
        rows, _, elapsed = run_camera(camera, collect, 24, 30)
    finally:
        camera.close()
    after = health()
    if len(rows) != len(retained):
        raise ValueError('Image retention incomplete')
    summary = summarize(rows, elapsed)
    out.mkdir(parents=True, exist_ok=False)
    for row, rgb in zip(rows, retained, strict=True):
        path = out/f'{row["index"]:04d}-raw.png'
        Image.fromarray(rgb).save(path)
        row.update(path=path.name, sha256=sha256(path))
    write_json(out/'report.json', dict(rows=rows, summary=summary, health_before=before, health_after=after,
        camera_config=str(config), requested_sensor_size=[2592,1944], processing_size=[320,240],
        camera_rotation_degrees=180, requested_camera_fps=5, queue=False, requested_frames=24,
        stopped_at_time_limit=len(rows)<24, sensor_clock='CLOCK_BOOTTIME',
        machine=platform.machine(), host=platform.node(), policy='head_blend',
        max_pre_camera_score_delta=max(d[0] for d in deltas), max_pre_camera_box_delta=max(d[1] for d in deltas),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        retained_rgb_bytes=sum(r.nbytes for r in retained),
        bundle_sha256=sha256(root/'bundle.json'), script_sha256=sha256(__file__),
        model_sha256=sha256(root/'models/classifier.tflite'),
        localizer_sha256=sha256(root/'models/head_localizer.tflite'),
        runtime_library_sha256=sha256(os.environ['DTR_TFLITE_LIBRARY']), region_library_sha256=sha256(native.path),
        scope='Single research pipeline, full camera capture plus processing. All processed RGBs saved after timing. Unlabeled live smoke test, not accuracy, comparative throughput, or flight qualification.',
        camera_used=True, model_replaced=False, accuracy_measured=False, deployment_approved=False,
        qualification_achieved=False, esp32_commands_sent=False, motor_commands_sent=False))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
