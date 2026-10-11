"""Sequential camera-cadence test; one model pipeline, no actuation or deployment."""
import argparse
import os
from pathlib import Path
import platform
import resource
import subprocess
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
from scripts.pi_peak_blend_live import summarize
from scripts.pi_red_blue_bench import health, statistics, verify_bundle

ORDER = (5, 10, 10, 5)


def telemetry():
    # Process names only, not arguments/environment that could contain credentials.
    return dict(health=health(), cpu_ticks=Path('/proc/stat').read_text().splitlines()[0],
        processes=subprocess.check_output(['ps', '-eo', 'pid,comm,pcpu,time'], timeout=3, text=True),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def aggregate(blocks):
    if tuple(b['requested_fps'] for b in blocks) != ORDER:
        raise ValueError('Require sequential ABBA block order')
    for block in blocks:
        if len(block['rows']) != 24 or summarize(block['rows'], block['summary']['elapsed_s']) != block['summary']:
            raise ValueError('Incomplete block or altered summary')
    rates = {}
    for fps in (5, 10):
        selected = [b for b in blocks if b['requested_fps'] == fps]
        rows = [r for b in selected for r in b['rows']]
        elapsed = sum(b['summary']['elapsed_s'] for b in selected)
        ages = [r['sensor_to_result_ms'] for r in rows if r['timestamp_valid']]
        rates[str(fps)] = dict(frames=len(rows), elapsed_s=elapsed, live_fps=len(rows)/elapsed,
            block_live_fps=[b['summary']['live_fps'] for b in selected],
            timing={k:statistics([r[k] for r in rows]) for k in (*TIMINGS, 'capture_and_convert_ms', 'frame_wall_ms')},
            sensor_to_result=statistics(ages) if ages else None,
            invalid_timestamp_frames=sum(not r['timestamp_valid'] for r in rows),
            increasing_sensor_timestamps=all(r['sensor_timestamp_increasing'] for r in rows),
            neural_calls=sum(r['neural_calls'] for r in rows),
            accepted_counts={c:sum(b['summary']['accepted_counts'][c] for b in selected)
                for c in ('red_balloon', 'blue_balloon')},
            exposure_ms=statistics([r['exposure_us']/1000 for r in rows]),
            frame_duration_ms=statistics([r['frame_duration_us']/1000 for r in rows]))
    healthy = all(b[k]['health']['throttled'] == 'throttled=0x0' for b in blocks for k in ('before', 'after'))
    valid = all(v['invalid_timestamp_frames'] == 0 and v['increasing_sensor_timestamps'] for v in rates.values())
    separated = min(rates['10']['block_live_fps']) > max(rates['5']['block_live_fps'])
    age_ok = valid and rates['10']['sensor_to_result']['p95_ms'] <= rates['5']['sensor_to_result']['p95_ms']
    return dict(rates=rates, no_throttle_flags=healthy, timestamps_valid=valid,
        faster_in_both_blocks=separated, p95_age_nonregressing=age_ok,
        eligible_for_further_10fps_research=healthy and valid and separated and age_ok)


def camera_block(factory, transform, infer, fps):
    if fps not in (5, 10):
        raise ValueError('Unsupported fixed camera rate')
    retained = []

    def collect(rgb):
        result = infer(rgb)
        retained.append(rgb)
        return result

    camera = factory()
    try:
        camera.configure(camera.create_video_configuration(
            main=dict(size=(320,240), format='RGB888'), sensor=dict(output_size=(2592,1944), bit_depth=10),
            controls=dict(FrameRate=fps), buffer_count=2, queue=False, transform=transform))
        config = camera.camera_configuration()
        validate_camera_config(config)
        camera.start()
        time.sleep(1)
        before = telemetry()
        rows, _, elapsed = run_camera(camera, collect, 24, 30)
        after = telemetry()
    finally:
        camera.close()
    if len(rows) != len(retained):
        raise ValueError('Missing retained RGB')
    return dict(requested_fps=fps, camera_config=str(config), rows=rows,
        summary=summarize(rows, elapsed), before=before, after=after), retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'library', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    verify_bundle(root)
    protocol = read_json(root/'camera-rates.json')
    if (tuple(protocol['order']) != ORDER or protocol['policy'] != 'head_blend'
            or protocol['frames_per_block'] != 24 or protocol['seconds_per_block'] != 30
            or protocol['settle_seconds'] != 1 or protocol['queue'] is not False
            or protocol['sensor_size'] != [2592,1944] or protocol['processing_size'] != [320,240]
            or protocol['rotation_degrees'] != 180):
        raise ValueError('Frozen camera-rate protocol changed')
    for key in ('DTR_RED_BLUE_COLOR_LOOKUP', 'DTR_RED_BLUE_LOOKUP_LIBRARY'):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    native = NativeBalloonRegions(args.library, direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(root/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    classifier = Predictor(root/'models/classifier.tflite', allow_unvalidated=True)
    network = Network(root/'models/head_localizer.tflite', True)
    if classifier.metadata['threshold'] != .8:
        raise ValueError('Threshold changed')

    def infer(rgb):
        return process(rgb, classifier, network, native, 'head_blend')

    frames = read_json(root/'decoder-frames.json')
    deltas = []
    for frame in frames[:3]:
        rgb = np.asarray(Image.open(root/frame['path']).convert('RGB'))
        deltas.append(compare(infer(rgb)['detections'], frame['expected']['head_blend']))
    if len(frames) != 141 or len(deltas) != 3:
        raise ValueError('Incomplete golden coverage')
    from libcamera import Transform
    from picamera2 import Picamera2
    blocks, images = [], []
    for index, fps in enumerate(ORDER):
        block, retained = camera_block(Picamera2, Transform(hflip=True, vflip=True), infer, fps)
        blocks.append(block)
        images.append(retained)
        print(dict(block=index, requested_fps=fps, summary=block['summary']), flush=True)
    comparison = aggregate(blocks)
    out.mkdir(parents=True, exist_ok=False)
    for index, (block, retained) in enumerate(zip(blocks, images, strict=True)):
        for row, rgb in zip(block['rows'], retained, strict=True):
            path = out/f'{index}-{row["index"]:04d}-raw.png'
            Image.fromarray(rgb).save(path)
            row.update(path=path.name, sha256=sha256(path))
    write_json(out/'report.json', dict(blocks=blocks, comparison=comparison,
        machine=platform.machine(), host=platform.node(), policy='head_blend',
        bundle_sha256=sha256(root/'bundle.json'), protocol_sha256=sha256(root/'camera-rates.json'),
        script_sha256=sha256(__file__), model_sha256=sha256(root/'models/classifier.tflite'),
        localizer_sha256=sha256(root/'models/head_localizer.tflite'),
        runtime_library_sha256=sha256(os.environ['DTR_TFLITE_LIBRARY']), region_library_sha256=sha256(native.path),
        max_pre_camera_score_delta=max(d[0] for d in deltas), max_pre_camera_box_delta=max(d[1] for d in deltas),
        retained_rgb_bytes=sum(rgb.nbytes for retained in images for rgb in retained),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope=protocol['scope'], camera_used=True, model_replaced=False, accuracy_measured=False,
        deployment_approved=False, qualification_achieved=False, esp32_commands_sent=False, motor_commands_sent=False))
    print(comparison, flush=True)


if __name__ == '__main__':
    main()
