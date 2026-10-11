"""Sequential current-head-blend/containment ABBA replay; no camera or motors."""
import argparse
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.pi_blue_peak_blend import Network, TIMINGS, process as baseline
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_red_blue_bench import health, statistics, verify_bundle

ORDER = ('parent', 'confidence', 'confidence', 'parent')


def process(rgb, classifier, network, native, policy):
    if policy not in ORDER:
        raise ValueError('Unknown containment policy')
    from scripts.current_containment_frozen import confidence_ordered
    result = baseline(rgb, classifier, network, native, 'head_blend')
    started = time.perf_counter()
    if policy == 'confidence':
        result['detections'] = confidence_ordered(result['detections'])
    extra = (time.perf_counter()-started)*1000
    result['processing_ms'] += extra
    result['selection_ms'] += extra
    result['containment_extra_ms'] = extra
    return result


def compare_all(actual, expected):
    delta = compare(actual, expected)
    if any(a.get('suppressed_by_index') != b.get('suppressed_by_index')
           for a, b in zip(actual, expected, strict=True)):
        raise ValueError('Containment winner differs')
    return delta


def replay(root, library, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_bundle(root)
    native = NativeBalloonRegions(library, direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(root/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    cv2.setNumThreads(1)
    for key in ('DTR_RED_BLUE_COLOR_LOOKUP', 'DTR_RED_BLUE_LOOKUP_LIBRARY'):
        os.environ.pop(key, None)
    classifier = Predictor(root/'models/classifier.tflite', allow_unvalidated=True)
    network = Network(root/'models/head_localizer.tflite', True)
    frames = read_json(root/'containment-frames.json')
    if len(frames) != 205 or classifier.metadata['threshold'] != .8:
        raise ValueError('Unexpected coverage/threshold')
    images = [np.asarray(Image.open(root/f['path']).convert('RGB')) for f in frames]
    for policy in ('parent', 'confidence'):
        for f, rgb in zip(frames[:3], images[:3], strict=True):
            compare_all(process(rgb, classifier, network, native, policy)['detections'], f['expected'][policy])
    blocks = []
    for index, policy in enumerate(ORDER):
        before, rows = health(), []
        for i, (frame, rgb) in enumerate(zip(frames, images, strict=True)):
            result = process(rgb, classifier, network, native, policy)
            score, box = compare_all(result['detections'], frame['expected'][policy])
            rows.append(dict({k: frame[k] for k in ('panel', 'id', 'source', 'truth', 'frame_sha256')},
                score_delta=score, box_delta=box, **result))
            if (i+1) % 40 == 0:
                print(f'Block{index+1}/4 {policy}: {i+1}/205', flush=True)
        blocks.append(dict(policy=policy, rows=rows, frames=len(rows), health_before=before, health_after=health(),
            timing={k: statistics([r[k] for r in rows]) for k in (*TIMINGS, 'containment_extra_ms')}))
    write_json(output, dict(blocks=blocks, machine=platform.machine(), host=platform.node(),
        bundle_sha256=sha256(root/'bundle.json'), script_sha256=sha256(__file__),
        classifier_sha256=sha256(root/'models/classifier.tflite'),
        localizer_sha256=sha256(root/'models/head_localizer.tflite'),
        runtime_library_sha256=sha256(os.environ['DTR_TFLITE_LIBRARY']), region_library_sha256=sha256(native.path),
        scope='Warmed ABBA205scene full-pipeline processing replay. One policy runs at a time; repeated timing rows are not independent scenes. Capture/display/serial excluded.',
        camera_used=False, model_replaced=False, esp32_commands_sent=False, motor_commands_sent=False,
        deployment_approved=False, qualification_achieved=False))
    print([dict(policy=b['policy'], timing=b['timing']) for b in blocks], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'library', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    replay(args.root, args.library, args.output)
