"""Selection-only microbenchmark with golden parity and counterbalanced blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from scripts.pi_balloon_search_bench import select_detections
from scripts.research_balloon_confidence_selection import confidence_selection


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(rows):
    return [{k: v for k, v in r.items() if k != 'latency_ms'} for r in rows]


def timing(values):
    ordered = sorted(values)
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
        p95_ms=ordered[int(.95*(len(ordered)-1))], max_ms=max(values))


def health():
    if platform.system() != 'Linux':
        return dict(throttled=None)
    return dict(throttled=subprocess.check_output(['vcgencmd', 'get_throttled'], text=True).strip(),
        temperature=subprocess.check_output(['vcgencmd', 'measure_temp'], text=True).strip())


def bench(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    bundle = json.loads((root/'bundle.json').read_text())
    for name, expected in bundle['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('Bundle integrity mismatch')
    inputs = json.loads((root/'inputs.json').read_text())
    functions = dict(current=lambda rows: select_detections(rows, 'split_merged', 2.42913421076193),
        confidence=confidence_selection)
    for row in inputs:
        for key, fn in functions.items():
            if stable(fn(row['found'])) != stable(row['expected'][key]):
                raise ValueError('Host/device selection parity failed')
    before = health()
    blocks = []
    # ABBA counterbalances monotonic thermal/drift effects; not simultaneous policies.
    for key in ('current', 'confidence', 'confidence', 'current'):
        fn = functions[key]
        samples = []
        start_cpu, start = time.process_time(), time.perf_counter()
        for _ in range(15):
            for row in inputs:
                tick = time.perf_counter_ns()
                fn(row['found'])
                samples.append((time.perf_counter_ns()-tick)/1e6)
        blocks.append(dict(policy=key, calls=len(samples), timing=timing(samples),
            wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-start_cpu))
    result = dict(machine=platform.machine(), host=platform.node(), python=platform.python_version(),
        bundle_sha256=digest(root/'bundle.json'), inputs_sha256=digest(root/'inputs.json'),
        script_sha256=digest(__file__), parity_rows=len(inputs), blocks=blocks,
        health_before=before, health_after=health(), camera_used=False, neural_inference_used=False,
        model_replaced=False, esp32_commands_sent=False, motor_commands_sent=False,
        scope='Selection-only warmed CPU microbenchmark on stored predictions; not full-pipeline FPS, live accuracy or qualification.')
    with output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    bench(args.root, args.output)
