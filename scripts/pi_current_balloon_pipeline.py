"""Isolated current research-pipeline replay, camera/serial/motors excluded."""
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
from scripts.pi_red_blue_bench import health, statistics, verify_bundle
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_balloon_search_fast import search_fast


def compare(actual, expected):
    if len(actual) != len(expected):
        raise ValueError('Proposal count differs from host')
    score_delta, box_delta = 0., 0.
    for a, b in zip(actual, expected, strict=True):
        for k in ('crop_box', 'label', 'accepted', 'raw_accepted', 'suppressed',
                  'suppressed_by', 'suppression_reason', 'rejection_reason'):
            if a.get(k) != b.get(k):
                raise ValueError('Host/device selection or crop mismatch: '+k)
        da = float(np.max(np.abs(np.asarray(a['scores'])-b['scores'])))
        db = float(np.max(np.abs(np.asarray(a['box'])-b['box'])))
        if not np.isfinite(da+db) or da > .001 or db > .02:
            raise ValueError('Host/device score or geometry mismatch')
        score_delta, box_delta = max(score_delta, da), max(box_delta, db)
    return score_delta, box_delta


class CoarseNetwork:
    def __init__(self, path):
        m = read_json(Path(path).with_suffix('.json'))
        if (m['kind'] != 'balloon_grid_convergence_research' or m['arm'] != 'coarse'
                or m['sha256'] != sha256(path) or m['deployment_approved'] is not False
                or m['proposal_threshold'] != .1 or m['max_added_blue_proposals'] != 2):
            raise ValueError('Unexpected proposal model')
        from dtr.native_tflite import Interpreter
        self.interpreter = Interpreter(path, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input, self.output = self.interpreter.get_input_details()[0], self.interpreter.get_output_details()[0]
        if (self.input['shape'].tolist() != [1, 120, 160, 3] or self.output['shape'].tolist() != [1, 15, 20, 6]
                or self.input['dtype'] != np.float32 or self.output['dtype'] != np.float32):
            raise ValueError('Unexpected localizer tensor contract')

    def predict(self, rgb):
        from scripts.current_dense_frozen import decode
        image = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)[None]
        self.interpreter.set_tensor(self.input['index'], image)
        self.interpreter.invoke()
        return decode(self.interpreter.get_tensor(self.output['index'])[0])


def process(rgb, classifier, localizer, native):
    from scripts.current_dense_frozen import supplement
    started = time.perf_counter()
    base = search_fast(rgb, native)
    searched = time.perf_counter()
    proposals = supplement(base, localizer.predict(rgb))
    localized = time.perf_counter()
    found = []
    for proposal in proposals:
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
    bundle = verify_bundle(root)
    native = NativeBalloonRegions(library, direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(root/'dtr/native_balloon_regions.c'):
        raise ValueError('Native reducer source differs')
    for key in ('DTR_RED_BLUE_COLOR_LOOKUP', 'DTR_RED_BLUE_LOOKUP_LIBRARY'):
        os.environ.pop(key, None)
    cv2.setNumThreads(1)
    classifier = Predictor(root/'models/classifier.tflite', allow_unvalidated=True)
    localizer = CoarseNetwork(root/'models/localizer.tflite')
    if classifier.metadata['threshold'] != .8:
        raise ValueError('Classifier operating point changed')
    frames = read_json(root/'frames.json')
    images = [np.asarray(Image.open(root/r['path']).convert('RGB')) for r in frames]
    for r, image in zip(frames[:3], images[:3], strict=True):
        compare(process(image, classifier, localizer, native)['detections'], r['expected'])
    before, rows = health(), []
    for index, (r, image) in enumerate(zip(frames, images, strict=True)):
        value = process(image, classifier, localizer, native)
        delta, geometry = compare(value['detections'], r['expected'])
        rows.append(dict(panel=r['panel'], id=r['id'], source=r['source'], frame_sha256=r['frame_sha256'],
            truth=r['truth'], score_delta=delta, box_delta=geometry, **value))
        if (index+1) % 20 == 0:
            print(f'Validated {index+1}/{len(frames)} frames', flush=True)
    report = dict(rows=rows, frames=len(rows), machine=platform.machine(),
        bundle_sha256=sha256(root/'bundle.json'), source_reports=bundle['source_reports'],
        script_sha256=sha256(__file__), model_sha256=sha256(root/'models/classifier.tflite'),
        localizer_sha256=sha256(root/'models/localizer.tflite'), region_library_sha256=sha256(native.path),
        runtime_library_sha256=sha256(os.environ['DTR_TFLITE_LIBRARY']),
        timing={key: statistics([r[key] for r in rows]) for key in
            ('processing_ms', 'search_ms', 'localizer_ms', 'classifier_ms', 'selection_ms')},
        processing_only_fps=1000/np.mean([r['processing_ms'] for r in rows]),
        max_score_delta=max(r['score_delta'] for r in rows), max_box_delta=max(r['box_delta'] for r in rows),
        health_before=before, health_after=health(), camera_used=False, deployment_approved=False,
        esp32_commands_sent=False, motor_commands_sent=False,
        scope='One research pipeline, warmed full-frame development replay; excludes capture and live scene accuracy. Not counterbalanced comparison or flight qualification.')
    write_json(output, report)
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'library', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    replay(args.root, args.library, args.output)
