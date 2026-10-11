"""ABBA full-pipeline replay of original and asymmetric search; no camera/actuation."""
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
from scripts.pi_current_balloon_pipeline import CoarseNetwork, compare
from scripts.pi_red_blue_bench import health, statistics, verify_bundle
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_balloon_search_fast import search_fast
from scripts.research_blue_bright_search import search


def process(rgb, classifier, localizer, native, policy):
    if policy not in ('original', 'blue_bright'):
        raise ValueError('Unknown search policy')
    from scripts.current_dense_frozen import supplement
    started = time.perf_counter()
    base = search_fast(rgb, native) if policy == 'original' else search(rgb, native, True)
    searched = time.perf_counter()
    proposed = supplement(base, localizer.predict(rgb))
    localized = time.perf_counter()
    found = []
    for proposal in proposed:
        a, b, c, d = proposal['crop_box']
        found.append(dict(proposal, **classifier.predict(rgb[b:d, a:c])))
    classified = time.perf_counter()
    selected = confidence_selection(found)
    done = time.perf_counter()
    return dict(detections=selected, neural_calls=len(found), localizer_calls=1,
        processing_ms=(done-started)*1000, search_ms=(searched-started)*1000,
        localizer_ms=(localized-searched)*1000, classifier_ms=(classified-localized)*1000,
        selection_ms=(done-classified)*1000)


def replay(root, library, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_bundle(root)
    native = NativeBalloonRegions(library, direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(root/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source mismatch')
    cv2.setNumThreads(1)
    for key in ('DTR_RED_BLUE_COLOR_LOOKUP', 'DTR_RED_BLUE_LOOKUP_LIBRARY'):
        os.environ.pop(key, None)
    classifier = Predictor(root/'models/classifier.tflite', allow_unvalidated=True)
    localizer = CoarseNetwork(root/'models/localizer.tflite')
    if classifier.metadata['threshold'] != .8:
        raise ValueError('Unexpected classifier threshold')
    frames = read_json(root/'paired-frames.json')
    if len(frames) != 141:
        raise ValueError('Require all141 development frames')
    images = [np.asarray(Image.open(root/r['path']).convert('RGB')) for r in frames]
    for policy in ('original', 'blue_bright'):
        for row, rgb in zip(frames[:3], images[:3], strict=True):
            compare(process(rgb, classifier, localizer, native, policy)['detections'], row['expected'][policy])
    blocks = []
    keys = ('processing_ms', 'search_ms', 'localizer_ms', 'classifier_ms', 'selection_ms')
    for block, policy in enumerate(('original', 'blue_bright', 'blue_bright', 'original')):
        before, rows = health(), []
        for index, (frame, rgb) in enumerate(zip(frames, images, strict=True)):
            result = process(rgb, classifier, localizer, native, policy)
            score, box = compare(result['detections'], frame['expected'][policy])
            rows.append(dict(panel=frame['panel'], id=frame['id'], source=frame['source'],
                frame_sha256=frame['frame_sha256'], truth=frame['truth'], score_delta=score, box_delta=box, **result))
            if (index+1) % 40 == 0:
                print(f'Block {block+1}/4 {policy}: {index+1}/141', flush=True)
        blocks.append(dict(policy=policy, rows=rows, frames=len(rows),
            timing={k: statistics([r[k] for r in rows]) for k in keys},
            health_before=before, health_after=health()))
    result = dict(blocks=blocks, machine=platform.machine(), host=platform.node(),
        bundle_sha256=sha256(root/'bundle.json'), script_sha256=sha256(__file__),
        model_sha256=sha256(root/'models/classifier.tflite'), localizer_sha256=sha256(root/'models/localizer.tflite'),
        runtime_library_sha256=sha256(os.environ['DTR_TFLITE_LIBRARY']), region_library_sha256=sha256(native.path),
        scope='Sequential ABBA warmed full-pipeline replay, same models/images. Capture, display and serial excluded; not independent scene accuracy.',
        camera_used=False, model_replaced=False, deployment_approved=False, qualification_achieved=False,
        esp32_commands_sent=False, motor_commands_sent=False)
    write_json(output, result)
    print({b['policy']+str(i): b['timing']['processing_ms'] for i, b in enumerate(blocks)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'library', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    replay(args.root, args.library, args.output)
