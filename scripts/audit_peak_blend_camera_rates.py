"""Audit every live cadence sample, timestamp, sensor period and host/Pi decision."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.balloon_dense_runtime import DenseProposals
from scripts.build_peak_blend_camera_rates import OUTPUT as ROOT, PROTOCOL
from scripts.pi_balloon_live_bench import frame_timing
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_peak_blend_camera_rates import aggregate
from scripts.pi_red_blue_bench import verify_bundle
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.research_predictor import classify_candidates

OUTPUT = Path('research/balloon-red-blue-20261008/peak-blend-camera-rates-20261011.json')


def cpu_activity(before, after):
    def parse(line):
        fields = line.split()
        if fields[0] != 'cpu' or len(fields) < 9:
            raise ValueError('Require aggregate Linux CPU counters')
        # user/nice already include guest time; do not double-count fields9/10.
        return np.asarray([int(v) for v in fields[1:9]], np.int64)

    delta = parse(after)-parse(before)
    if np.any(delta < 0) or delta.sum() <= 0:
        raise ValueError('CPU counters did not advance monotonically')
    idle = int(delta[3]+delta[4])
    total = int(delta.sum())
    return dict(total_ticks=total, idle_and_iowait_ticks=idle,
        busy_percent=100*(total-idle)/total,
        scope='System-wide counter interval including boundary telemetry; not model-only CPU time. I/O wait treated as nonbusy; guest time not counted twice.')


def validate_clock_and_rate(rows, fps):
    previous = None
    for row in rows:
        metadata = dict(SensorTimestamp=row['sensor_timestamp_ns'], ExposureTime=row['exposure_us'],
            AnalogueGain=row['analogue_gain'], FrameDuration=row['frame_duration_us'])
        timing = frame_timing(metadata, row['capture_started_boottime_ns'], row['capture_completed_boottime_ns'],
            row['result_boottime_ns'], previous)
        if any(row[k] != v for k,v in timing.items()) or not timing['timestamp_valid'] or not timing['sensor_timestamp_increasing']:
            raise ValueError('Camera timestamp validation failed')
        previous = row['sensor_timestamp_ns']
        if (not isinstance(row['frame_duration_us'], (float,int))
                or not np.isfinite(row['frame_duration_us'])
                or abs(row['frame_duration_us']-1e6/fps) > .05*1e6/fps):
            raise ValueError('Sensor did not apply the requested cadence within5%')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    bundle = verify_bundle(ROOT)
    path = ROOT/'camera/report.json'
    report = read_json(path)
    if (report['machine'] != 'armv6l' or report['policy'] != 'head_blend'
            or report['bundle_sha256'] != sha256(ROOT/'bundle.json')
            or report['protocol_sha256'] != sha256(PROTOCOL) or bundle['protocol_sha256'] != sha256(PROTOCOL)
            or report['script_sha256'] != sha256('scripts/pi_peak_blend_camera_rates.py')
            or report['model_sha256'] != sha256(ROOT/'models/classifier.tflite')
            or report['localizer_sha256'] != sha256(ROOT/'models/head_localizer.tflite')
            or report['runtime_library_sha256'] != '54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a'
            or report['region_library_sha256'] != 'd7620537d7e9850f0f794a26facc188063f289c6ad9f5d793693ad5715e43303'
            or report['camera_used'] is not True or report['retained_rgb_bytes'] != 96*240*320*3
            or any(report[k] is not False for k in ('model_replaced', 'accuracy_measured', 'deployment_approved',
                'qualification_achieved', 'esp32_commands_sent', 'motor_commands_sent'))):
        raise ValueError('Live comparison identity or scope changed')
    comparison = aggregate(report['blocks'])
    if comparison != report['comparison']:
        raise ValueError('All-sample comparison does not reproduce')
    classifier = Predictor(ROOT/'models/classifier.tflite', allow_unvalidated=True)
    network = DenseProposals(ROOT/'models/head_localizer.tflite')
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(ROOT/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    deltas, files, blocks = [], {}, []
    for index, block in enumerate(report['blocks']):
        validate_clock_and_rate(block['rows'], block['requested_fps'])
        for row in block['rows']:
            if row['path'] != f'{index}-{row["index"]:04d}-raw.png':
                raise ValueError('Frame identity changed')
            image = path.parent/row['path']
            if sha256(image) != row['sha256']:
                raise ValueError('Retained image changed')
            files[row['path']] = row['sha256']
            rgb = np.asarray(Image.open(image).convert('RGB'))
            if rgb.shape != (240,320,3):
                raise ValueError('Pixel size changed')
            base = search(rgb, native, True)
            network.predict(rgb)
            grid = network.interpreter.get_tensor(network.output['index'])[0]
            proposals = supplement(base, decode(grid, True))
            detections = confidence_selection(classify_candidates(classifier, rgb, proposals))
            deltas.append(compare(detections, row['detections']))
        blocks.append(dict({k:block[k] for k in ('requested_fps', 'summary', 'before', 'after', 'camera_config')},
            cpu_activity=cpu_activity(block['before']['cpu_ticks'], block['after']['cpu_ticks'])))
    write_json(OUTPUT, dict(comparison=comparison, blocks=blocks, host_pi_parity_frames=len(deltas),
        max_score_delta=max(d[0] for d in deltas), max_box_delta=max(d[1] for d in deltas),
        raw=dict(path=str(path), sha256=sha256(path)), retained_frames=files,
        bundle_sha256=sha256(ROOT/'bundle.json'), protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        max_rss_kib=report['max_rss_kib'], retained_rgb_bytes=report['retained_rgb_bytes'],
        scope='Sequential ABBA camera-inclusive timing in unlabeled live views. Sensor periods validated within5%of requested cadence. All startup and outlier samples retained. Not matched-scene accuracy or independent qualification.',
        camera_used=True, accuracy_measured=False, model_replaced=False, deployment_approved=False,
        qualification_achieved=False, esp32_commands_sent=False, motor_commands_sent=False))
    print(dict(comparison=comparison, parity_frames=len(deltas), max_score_delta=max(d[0] for d in deltas)))


if __name__ == '__main__':
    audit()
