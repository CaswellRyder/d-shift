"""Sequential paired ABBA timing with native exact and cross-host tolerance checks."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from scripts.pi_decoder_cost_probe import digest, compare, health
from scripts.research_blue_peak_blend import decode as frozen
from scripts.independent_neighbor_decoder import decode as neighbor
from scripts.cached_balloon_decoder import decode as cached

ROUNDS = ['frozen', 'cached_frozen', 'cached_frozen', 'frozen',
          'neighbor', 'cached_neighbor', 'cached_neighbor', 'neighbor']


def run(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Preserve probe receipt')
    manifest = json.loads((root/'manifest.json').read_text())
    for name, h in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path) != h:
            raise ValueError('Probe identity changed')
    values = np.load(root/'outputs.npy', allow_pickle=False)
    if values.shape != (42, 15, 20, 6) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Unexpected saved model outputs')
    expected = json.loads((root/'expected.json').read_text())
    methods = dict(frozen=lambda v: frozen(v, True), neighbor=neighbor,
                   cached_frozen=lambda v: cached(v, False), cached_neighbor=lambda v: cached(v, True))
    before, started = health(), time.monotonic()
    parity, native = {}, {}
    for name, reference in [('frozen', 'frozen'), ('neighbor', 'independent_neighbor')]:
        native[name] = [methods[name](v) for v in values]
        parity[name] = max(compare(a, b) for a, b in zip(native[name], expected[reference], strict=True))
        if [methods['cached_'+name](v) for v in values] != native[name]:
            raise ValueError('Cached decoder differs from native reference')
    timings = []
    for round_index, name in enumerate(ROUNDS):
        reference = name.removeprefix('cached_')
        for index, value in enumerate(values):
            if time.monotonic()-started > 120:
                raise TimeoutError('Bounded replay exceeded 120 seconds')
            start = time.perf_counter_ns()
            result = methods[name](value)
            elapsed = (time.perf_counter_ns()-start)/1e6
            if result != native[reference][index]:
                raise ValueError('Timed decoder result differs')
            timings.append(dict(round=round_index, method=name, index=index, elapsed_ms=elapsed,
                candidates=len(result), rescues=sum(r.get('neighbor_rescue', False) for r in result)))
    summary = {}
    for name in methods:
        ms = np.asarray([r['elapsed_ms'] for r in timings if r['method'] == name])
        summary[name] = dict(calls=len(ms), mean_ms=float(ms.mean()), p50_ms=float(np.percentile(ms, 50)),
            p95_ms=float(np.percentile(ms, 95)), max_ms=float(ms.max()))
    report = dict(machine=platform.machine(), host=platform.node(), numpy=np.__version__,
        manifest_sha256=digest(root/'manifest.json'), script_sha256=digest(__file__),
        native_outputs_exact=True, cross_host_max_box_delta=parity, timings=timings, summary=summary,
        health_before=before, health_after=health(), elapsed_s=time.monotonic()-started,
        scope='Two sequential ABBA decoder-only pairs on 42 saved TRAIN output grids. No inference, capture, tracking, transport or controller timing. Process snapshots cannot establish whole-system isolation. No FPS or accuracy qualification.',
        both_methods_simultaneous=False, inference_tested=False, camera_tested=False,
        motors_commanded=False, esp32_commanded=False, active_model_changed=False,
        development_evaluated=False, test_evaluated=False, deployment_approved=False, qualification_achieved=False)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(machine=report['machine'], summary=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.base, args.output)
