"""Recompute live timing and replay every retained camera frame on the host."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.balloon_dense_runtime import DenseProposals
from scripts.build_peak_blend_live import OUTPUT as ROOT
from scripts.pi_balloon_live_bench import frame_timing
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_peak_blend_live import summarize
from scripts.pi_red_blue_bench import verify_bundle
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.research_predictor import classify_candidates

OUTPUT = Path('research/balloon-red-blue-20261008/peak-blend-live-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    bundle = verify_bundle(ROOT)
    path = ROOT/'camera/report.json'
    report = read_json(path)
    if (report['machine'] != 'armv6l' or report['policy'] != 'head_blend'
            or report['bundle_sha256'] != sha256(ROOT/'bundle.json')
            or report['script_sha256'] != sha256('scripts/pi_peak_blend_live.py')
            or report['model_sha256'] != sha256(ROOT/'models/classifier.tflite')
            or report['localizer_sha256'] != sha256(ROOT/'models/head_localizer.tflite')
            or report['runtime_library_sha256'] != '54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a'
            or report['region_library_sha256'] != 'd7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303'
            or report['camera_rotation_degrees'] != 180 or report['requested_sensor_size'] != [2592,1944]
            or report['processing_size'] != [320,240] or report['queue'] is not False
            or report['requested_camera_fps'] != bundle['protocol']['fps']
            or report['camera_used'] is not True
            or any(report[k] is not False for k in ('model_replaced', 'accuracy_measured', 'deployment_approved',
                'qualification_achieved', 'esp32_commands_sent', 'motor_commands_sent'))):
        raise ValueError('Live identity or scope changed')
    rows = report['rows']
    if (len(rows) != 24 or report['stopped_at_time_limit'] or report['retained_rgb_bytes'] != 24*240*320*3
            or summarize(rows, report['summary']['elapsed_s']) != report['summary']):
        raise ValueError('Incomplete camera coverage or summary mismatch')
    classifier = Predictor(ROOT/'models/classifier.tflite', allow_unvalidated=True)
    network = DenseProposals(ROOT/'models/head_localizer.tflite')
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(ROOT/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source mismatch')
    previous, deltas, files = None, [], {}
    for row in rows:
        if row['path'] != f'{row["index"]:04d}-raw.png':
            raise ValueError('Frame path changed')
        image = path.parent/row['path']
        if sha256(image) != row['sha256']:
            raise ValueError('Retained frame changed')
        files[row['path']] = row['sha256']
        metadata = dict(SensorTimestamp=row['sensor_timestamp_ns'], ExposureTime=row['exposure_us'],
            AnalogueGain=row['analogue_gain'], FrameDuration=row['frame_duration_us'])
        timing = frame_timing(metadata, row['capture_started_boottime_ns'], row['capture_completed_boottime_ns'],
            row['result_boottime_ns'], previous)
        if any(row[k] != v for k,v in timing.items()) or not timing['timestamp_valid'] or not timing['sensor_timestamp_increasing']:
            raise ValueError('Camera timestamp validation failed')
        previous = row['sensor_timestamp_ns']
        rgb = np.asarray(Image.open(image).convert('RGB'))
        if rgb.shape != (240,320,3):
            raise ValueError('Camera pixels changed size')
        base = search(rgb, native, True)
        network.predict(rgb)
        grid = network.interpreter.get_tensor(network.output['index'])[0]
        proposals = supplement(base, decode(grid, True))
        detections = confidence_selection(classify_candidates(classifier, rgb, proposals))
        deltas.append(compare(detections, row['detections']))
    write_json(OUTPUT, dict(summary=report['summary'], max_score_delta=max(d[0] for d in deltas),
        max_box_delta=max(d[1] for d in deltas), host_pi_parity_frames=len(deltas), retained_frames=files,
        raw=dict(path=str(path), sha256=sha256(path)), bundle_sha256=sha256(ROOT/'bundle.json'),
        script_sha256=sha256(__file__), health_before=report['health_before'], health_after=report['health_after'],
        max_rss_kib=report['max_rss_kib'], retained_rgb_bytes=report['retained_rgb_bytes'],
        scope='Full candidate live camera smoke and host parity. Unlabeled scene; accepted detections are not true positives. No accuracy, motion robustness or flight qualification claim.',
        camera_used=True, accuracy_measured=False, model_replaced=False, deployment_approved=False,
        qualification_achieved=False, esp32_commands_sent=False, motor_commands_sent=False))
    print(read_json(OUTPUT))


if __name__ == '__main__':
    audit()
