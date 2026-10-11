"""Opt-in complete gated-selection replay. No camera access or actuation."""
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
from scripts.pi_balloon_search_bench import check_predictions, select_detections
from scripts.pi_red_blue_bench import health, verify_bundle
from scripts.research_balloon_search import experimental
from scripts.research_balloon_search_fast import search_fast
from scripts.research_dense_supported_selection import candidate_pairs, supported_selection


class GuardedDense:
    """At most one failed call per process, then baseline selection only."""
    def __init__(self, network):
        self.network = network
        self.error = None

    def predict(self, rgb):
        if self.error is not None:
            return []
        try:
            return self.network.predict(rgb)
        except (RuntimeError, ValueError, OSError) as error:
            self.error = type(error).__name__ + ': ' + str(error)
            return []


class BundledDense:
    def __init__(self, path, decoder_identity):
        metadata = read_json(Path(path).with_suffix('.json'))
        if (metadata['sha256'] != sha256(path)
                or metadata['kind'] != 'balloon_dense_proposal_research'
                or metadata['decoder_sha256'] != decoder_identity
                or metadata['proposal_threshold'] != .1 or metadata['max_proposals'] != 12
                or metadata['deployment_approved'] is not False):
            raise ValueError('Dense model contract changed')
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                from dtr.native_tflite import Interpreter
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if (self.input['shape'].tolist() != [1,120,160,3]
                or self.output['shape'].tolist() != [1,15,20,6]
                or self.input['dtype'] != np.float32 or self.output['dtype'] != np.float32):
            raise ValueError('Unexpected dense tensor contract')

    def predict(self, rgb):
        # Generated verbatim from the hash-pinned canonical decoder by the builder.
        from scripts.dense_decode_frozen import decode
        values = cv2.resize(rgb, (160,120), interpolation=cv2.INTER_AREA).astype(np.float32)[None]
        self.interpreter.set_tensor(self.input['index'], values)
        self.interpreter.invoke()
        return decode(self.interpreter.get_tensor(self.output['index'])[0])


def process(rgb, predictor, dense, native=None):
    if rgb.shape != (240,320,3) or rgb.dtype != np.uint8:
        raise ValueError('Require 320x240 RGB uint8')
    started = time.perf_counter()
    proposals = search_fast(rgb, native) if native is not None else experimental(rgb, 'mser')
    raw = []
    for proposal in proposals:
        a,b,c,d = proposal['crop_box']
        prediction = predictor.predict(rgb[b:d,a:c])
        raw.append(dict(proposal, **{k:prediction[k] for k in ('label','scores','score','accepted')}))
    baseline = select_detections(raw, 'split_merged', 2.42913421076193)
    gated = bool(candidate_pairs(raw))
    called = int(gated and dense.error is None)
    start_dense = time.perf_counter()
    learned = dense.predict(rgb) if called else []
    selected = supported_selection(raw, learned)
    return dict(detections=selected, baseline=baseline, neural_calls=len(raw),
        dense_calls=called, dense_gate=gated, dense_error=dense.error,
        dense_and_selection_ms=(time.perf_counter()-start_dense)*1000,
        processing_ms=(time.perf_counter()-started)*1000)


def compare(actual, expected):
    delta = check_predictions(actual, expected)
    for a,b in zip(actual, expected, strict=True):
        for key in ('suppression_reason','dense_supported_child'):
            if a.get(key) != b.get(key):
                raise ValueError('Selection reason changed')
    return delta


def replay(root, seed, library, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    bundle = verify_bundle(root)
    native = NativeBalloonRegions(library, direct=True)
    native_receipt = read_json(native.path.with_suffix('.json'))
    if native_receipt['source_sha256'] != sha256(root/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    for name in ('DTR_RED_BLUE_COLOR_LOOKUP','DTR_RED_BLUE_LOOKUP_LIBRARY'):
        os.environ.pop(name, None)
    cv2.setNumThreads(1)
    classifier = root/'models/classifier.tflite'
    predictor = Predictor(classifier, allow_unvalidated=True)
    dense_path = root/f'models/dense-{seed}.tflite'
    dense = GuardedDense(BundledDense(dense_path, bundle['decoder_original_sha256']))
    frames, rows, maximum = read_json(root/'frames.json'), [], 0.
    before = health()
    for row in frames:
        with Image.open(root/row['path']) as image:
            rgb = np.asarray(image.convert('RGB'))
        observed = process(rgb, predictor, dense, native)
        maximum = max(maximum, compare(observed['baseline'],row['baseline']),
                      compare(observed['detections'],row['selected'][seed]))
        if observed['dense_error'] or observed['dense_calls'] != row['dense_calls']:
            raise ValueError('Dense failure or gate mismatch: '+str(observed['dense_error']))
        rows.append(dict(panel=row['panel'], id=row['id'], **observed))
    result = dict(machine=platform.machine(), seed=seed, frames=len(rows),
        dense_calls=sum(r['dense_calls'] for r in rows), rows=rows,
        max_host_score_delta=maximum, full_pipeline_parity=True,
        classifier_sha256=sha256(classifier), dense_sha256=sha256(dense_path),
        bundle_sha256=sha256(root/'bundle.json'), region_library_sha256=sha256(native.path),
        region_build_receipt=native_receipt,
        runtime=type(predictor.interpreter).__module__,
        runtime_library_sha256=sha256(os.environ['DTR_TFLITE_LIBRARY']) if os.environ.get('DTR_TFLITE_LIBRARY') else None,
        opencv=cv2.__version__, health_before=before, health_after=health(),
        scope='Full search, crop classification, gated localizer and selection replay; contended, not isolated FPS or camera accuracy.',
        camera_used=False, isolated_speed_measured=False, deployment_approved=False)
    write_json(output, result)
    print({k:v for k,v in result.items() if k!='rows'}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','region-library','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--seed', choices=('47','48'), required=True)
    a = p.parse_args()
    replay(a.base, a.seed, a.region_library, a.output)
