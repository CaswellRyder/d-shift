"""Sequential ABCCBA full-pipeline decoder replay; no camera, serial or actuation."""
import argparse
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json,sha256,write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.pi_current_balloon_pipeline import compare
from scripts.pi_red_blue_bench import health,statistics,verify_bundle
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode as blended_decode

ORDER = ('original','blend','head_blend','head_blend','blend','original')
TIMINGS = ('processing_ms','search_ms','localizer_ms','classifier_ms','selection_ms')


class Network:
    def __init__(self,path,head=False):
        m = read_json(Path(path).with_suffix('.json'))
        if (m['sha256'] != sha256(path) or m['deployment_approved'] is not False
                or m['proposal_threshold'] != .1 or m['max_added_blue_proposals'] != 2
                or m['max_proposals'] != 12
                or (head and (m['kind'] != 'balloon_dense_proposal_research' or m['experiment'] != 'blue_head_only'))
                or (not head and (m['kind'] != 'balloon_grid_convergence_research' or m['arm'] != 'coarse'))):
            raise ValueError('Unexpected frozen model')
        from dtr.native_tflite import Interpreter
        self.interpreter = Interpreter(path,num_threads=1)
        self.interpreter.allocate_tensors()
        self.input,self.output = self.interpreter.get_input_details()[0],self.interpreter.get_output_details()[0]
        if (self.input['shape'].tolist() != [1,120,160,3] or self.output['shape'].tolist() != [1,15,20,6]
                or self.input['dtype'] != np.float32 or self.output['dtype'] != np.float32):
            raise ValueError('Localizer tensor contract changed')

    def predict(self,rgb,blend):
        from scripts.current_dense_frozen import decode
        if rgb.shape != (240,320,3) or rgb.dtype != np.uint8:
            raise ValueError('Require320x240 RGB uint8')
        image = cv2.resize(rgb,(160,120),interpolation=cv2.INTER_AREA).astype(np.float32)[None]
        self.interpreter.set_tensor(self.input['index'],image)
        self.interpreter.invoke()
        grid = self.interpreter.get_tensor(self.output['index'])[0]
        return blended_decode(grid,True) if blend else decode(grid)


def process(rgb,classifier,network,native,policy):
    if policy not in ORDER:
        raise ValueError('Unknown policy')
    from scripts.current_dense_frozen import supplement
    start = time.perf_counter()
    base = search(rgb,native,True)
    searched = time.perf_counter()
    proposed = supplement(base,network.predict(rgb,policy != 'original'))
    localized = time.perf_counter()
    found = []
    for proposal in proposed:
        a,b,c,d = proposal['crop_box']
        found.append(dict(proposal,**classifier.predict(rgb[b:d,a:c])))
    classified = time.perf_counter()
    ds = confidence_selection(found)
    done = time.perf_counter()
    return dict(detections=ds,neural_calls=len(found),localizer_calls=1,
        processing_ms=(done-start)*1000,search_ms=(searched-start)*1000,
        localizer_ms=(localized-searched)*1000,classifier_ms=(classified-localized)*1000,
        selection_ms=(done-classified)*1000)


def replay(root,library,output):
    root,output = Path(root).resolve(),Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_bundle(root)
    native = NativeBalloonRegions(library,direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(root/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source mismatch')
    cv2.setNumThreads(1)
    for key in ('DTR_RED_BLUE_COLOR_LOOKUP','DTR_RED_BLUE_LOOKUP_LIBRARY'):
        os.environ.pop(key,None)
    classifier = Predictor(root/'models/classifier.tflite',allow_unvalidated=True)
    networks = dict(parent=Network(root/'models/localizer.tflite'),head=Network(root/'models/head_localizer.tflite',True))
    frames = read_json(root/'decoder-frames.json')
    if len(frames) != 141 or classifier.metadata['threshold'] != .8:
        raise ValueError('Unexpected replay coverage or threshold')
    images = [np.asarray(Image.open(root/r['path']).convert('RGB')) for r in frames]
    for policy in ('original','blend','head_blend'):
        network = networks['head' if policy == 'head_blend' else 'parent']
        for row,rgb in zip(frames[:3],images[:3],strict=True):
            compare(process(rgb,classifier,network,native,policy)['detections'],row['expected'][policy])
    blocks = []
    for block,policy in enumerate(ORDER):
        network = networks['head' if policy == 'head_blend' else 'parent']
        before,rows = health(),[]
        for index,(frame,rgb) in enumerate(zip(frames,images,strict=True)):
            result = process(rgb,classifier,network,native,policy)
            score,box = compare(result['detections'],frame['expected'][policy])
            rows.append(dict({k:frame[k] for k in ('panel','id','source','frame_sha256','truth')},
                score_delta=score,box_delta=box,**result))
            if (index+1)%40 == 0:
                print(f'Block{block+1}/6 {policy}: {index+1}/141',flush=True)
        blocks.append(dict(policy=policy,rows=rows,frames=len(rows),
            timing={k:statistics([r[k] for r in rows]) for k in TIMINGS},health_before=before,health_after=health()))
    write_json(output,dict(blocks=blocks,machine=platform.machine(),host=platform.node(),
        bundle_sha256=sha256(root/'bundle.json'),script_sha256=sha256(__file__),
        model_sha256=sha256(root/'models/classifier.tflite'),localizer_sha256=sha256(root/'models/localizer.tflite'),
        head_localizer_sha256=sha256(root/'models/head_localizer.tflite'),
        runtime_library_sha256=sha256(os.environ['DTR_TFLITE_LIBRARY']),region_library_sha256=sha256(native.path),
        scope='Sequential ABCCBA warmed full-pipeline replay; one policy runs at a time. Excludes camera, display and serial. Repetitions are timing samples, not independent scenes.',
        camera_used=False,model_replaced=False,deployment_approved=False,qualification_achieved=False,
        esp32_commands_sent=False,motor_commands_sent=False))
    print({f'{i}/{b["policy"]}':b['timing']['processing_ms'] for i,b in enumerate(blocks)},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','library','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    replay(args.root,args.library,args.output)
