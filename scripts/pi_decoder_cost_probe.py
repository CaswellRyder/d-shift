"""Bounded ABBA decoder-only replay; no inference, camera or actuator access."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from scripts.research_blue_peak_blend import decode as frozen
from scripts.independent_neighbor_decoder import decode as neighbor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected):
    if len(actual) != len(expected):
        raise ValueError('Cross-host decoder candidate count differs')
    delta = 0.
    for a, b in zip(actual, expected, strict=True):
        for key in ('crop_box', 'color_group'):
            if a[key] != b[key]:
                raise ValueError('Cross-host discrete geometry differs')
        if a.get('neighbor_rescue', False) != b.get('neighbor_rescue', False):
            raise ValueError('Rescue decision differs')
        np.testing.assert_allclose(a['box'], b['box'], atol=.001, rtol=0)
        np.testing.assert_allclose(a['proposal_score'], b['proposal_score'], atol=1e-5, rtol=0)
        delta = max(delta, float(np.max(np.abs(np.asarray(a['box'])-b['box']))))
    return delta


def health():
    found = {}
    for key, path in (('temperature_millicelsius', '/sys/class/thermal/thermal_zone0/temp'),
                      ('load_average', '/proc/loadavg')):
        p = Path(path)
        found[key] = p.read_text().strip() if p.exists() else None
    for key, command in (('throttled', ['vcgencmd', 'get_throttled']),
                         ('process_snapshot', ['ps', '-eo', 'pid,comm,pcpu'])):
        try:
            found[key] = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            found[key] = None
    return found


def run(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Preserve probe receipt')
    manifest = json.loads((root/'manifest.json').read_text())
    for name, h in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path) != h:
            raise ValueError('Probe source or tensor identity changed')
    values = np.load(root/'outputs.npy', allow_pickle=False)
    if values.shape != (42, 15, 20, 6) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Unexpected saved model outputs')
    expected = json.loads((root/'expected.json').read_text())
    methods = {'frozen': lambda v: frozen(v, True), 'independent_neighbor': neighbor}
    before, started = health(), time.monotonic()
    parity = {}
    for name, decode in methods.items():
        parity[name] = max(compare(decode(v), e) for v, e in zip(values, expected[name], strict=True))
    timings = []
    for round_index, name in enumerate(('frozen', 'independent_neighbor', 'independent_neighbor', 'frozen')):
        for index, v in enumerate(values):
            if time.monotonic()-started > 120:
                raise TimeoutError('Bounded replay exceeded120seconds; no complete timing claim')
            start = time.perf_counter_ns()
            result = methods[name](v)
            milliseconds = (time.perf_counter_ns()-start)/1e6
            compare(result, expected[name][index])
            timings.append(dict(round=round_index, method=name, index=index,
                elapsed_ms=milliseconds, candidates=len(result), rescues=sum(r.get('neighbor_rescue', False) for r in result)))
    summary = {}
    for name in methods:
        ms = np.asarray([r['elapsed_ms'] for r in timings if r['method'] == name])
        summary[name] = dict(calls=len(ms), mean_ms=float(ms.mean()), p50_ms=float(np.percentile(ms, 50)),
            p95_ms=float(np.percentile(ms, 95)), max_ms=float(ms.max()))
    report = dict(machine=platform.machine(), host=platform.node(), numpy=np.__version__,
        manifest_sha256=digest(root/'manifest.json'), script_sha256=digest(__file__),
        parity_max_box_delta=parity, timings=timings, summary=summary,
        mean_added_ms=summary['independent_neighbor']['mean_ms']-summary['frozen']['mean_ms'],
        health_before=before, health_after=health(), elapsed_s=time.monotonic()-started,
        scope='Sequential ABBA decoder-only timing on42saved TRAIN output grids from two fixed models,84calls per method. Excludes inference, capture, transport, tracking and controller work. Process snapshots are not proof of whole-system isolation. Not end-to-end FPS or accuracy qualification.',
        both_methods_simultaneous=False, inference_tested=False, camera_tested=False,
        motors_commanded=False, esp32_commanded=False, active_model_changed=False,
        development_evaluated=False, test_evaluated=False, deployment_approved=False, qualification_achieved=False)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(machine=report['machine'], summary=summary, added_ms=report['mean_added_ms'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.base, args.output)
