"""Bounded tensor-only compatibility probe; not camera or isolated FPS evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('Probe bundle changed or escaped root')
    return manifest


def run(root, output):
    from native_tflite import Interpreter
    root, output = Path(root), Path(output)
    manifest = verified(root)
    output.mkdir(exist_ok=False)
    images = np.load(root / 'inputs.npy', allow_pickle=False)
    if images.shape != (manifest['frames'], 120, 160, 3) or images.dtype != np.float32:
        raise ValueError('Input contract changed')
    report = dict(machine=platform.machine(), host=platform.node(),
                  manifest_sha256=digest(root / 'manifest.json'), models={},
                  scope='Tensor inference only with competing camera workload; NOT isolated speed or live accuracy.',
                  camera_tested=False, isolated=False, deployment_approved=False)
    for seed, filename in manifest['models'].items():
        interpreter = Interpreter(str(root / filename), num_threads=1)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        if (inp['shape'].tolist() != [1,120,160,3] or out['shape'].tolist() != [1,15,20,6]
                or inp['dtype'] != np.float32 or out['dtype'] != np.float32):
            raise ValueError('Tensor contract changed')
        predictions, elapsed = [], []
        for rgb in images:
            start = time.perf_counter()
            interpreter.set_tensor(inp['index'], rgb[None])
            interpreter.invoke()
            predictions.append(interpreter.get_tensor(out['index'])[0])
            elapsed.append((time.perf_counter()-start)*1000)
        path = output / ('predictions-'+seed+'.npy')
        np.save(path, np.asarray(predictions), allow_pickle=False)
        report['models'][seed] = dict(model_sha256=digest(root / filename),
            output_sha256=digest(path), runtime_version=interpreter.version,
            tensor_call_ms=elapsed, calls=len(elapsed))
        interpreter.close()
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({s:r['calls'] for s,r in report['models'].items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.base, a.output)
