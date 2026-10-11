"""Independently recount immutable decoder probes; never infer full-system FPS."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json

OUT = Path('research/balloon-red-blue-20261008/decoder-cost-probes-20261011.json')
BASE_ROUNDS = ['frozen', 'independent_neighbor', 'independent_neighbor', 'frozen']
FAST_ROUNDS = ['frozen', 'cached_frozen', 'cached_frozen', 'frozen',
               'neighbor', 'cached_neighbor', 'cached_neighbor', 'neighbor']


def recount(report, expected, rounds):
    if len(report['timings']) != 42*len(rounds):
        raise ValueError('Missing or extra timing rows')
    for position, row in enumerate(report['timings']):
        round_index, index = divmod(position, 42)
        method = rounds[round_index]
        if (row['round'], row['index'], row['method']) != (round_index, index, method):
            raise ValueError('Timing order changed')
        reference = method.removeprefix('cached_')
        reference = 'independent_neighbor' if reference == 'neighbor' else reference
        predictions = expected[reference][index]
        if row['candidates'] != len(predictions) or row['rescues'] != sum(r.get('neighbor_rescue', False) for r in predictions):
            raise ValueError('Timing candidate identity changed')
        if not np.isfinite(row['elapsed_ms']) or row['elapsed_ms'] <= 0:
            raise ValueError('Invalid duration')
    summary = {}
    for method in dict.fromkeys(rounds):
        ms = np.array([r['elapsed_ms'] for r in report['timings'] if r['method'] == method])
        result = dict(calls=len(ms), mean_ms=float(ms.mean()), p50_ms=float(np.percentile(ms, 50)),
            p95_ms=float(np.percentile(ms, 95)), max_ms=float(ms.max()))
        if report['summary'][method]['calls'] != 84:
            raise ValueError('Wrong calls per method')
        for key, value in result.items():
            np.testing.assert_allclose(report['summary'][method][key], value, atol=1e-9, rtol=0)
        summary[method] = result
    return summary


def verify_bundle(root):
    manifest = read_json(root/'manifest.json')
    for name, digest in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError('Bundle identity changed')
    for name, digest in manifest['source_files'].items():
        if sha256(name) != digest:
            raise ValueError('Probe sources changed after measurement')
    if len(manifest['frames']) != 42 or {r['seed'] for r in manifest['frames']} != {47, 48}:
        raise ValueError('Wrong TRAIN tensor coverage')
    return manifest


def audit():
    if OUT.exists():
        raise FileExistsError('Preserve research receipt')
    inputs, results = {}, {}
    for variant in ('decoder-cost', 'cached-decoder'):
        root = Path(f'artifacts/{variant}-probe-20261011')
        manifest = verify_bundle(root)
        inputs[str(root/'manifest.json')] = sha256(root/'manifest.json')
        inputs.update(manifest['source_files'])
        if variant == 'cached-decoder':
            if manifest['source_manifest_sha256'] != sha256('artifacts/decoder-cost-probe-20261011/manifest.json'):
                raise ValueError('Cached probe did not reuse fixed sample')
            for name in ('outputs.npy', 'expected.json'):
                if manifest['files'][name] != sha256(Path('artifacts/decoder-cost-probe-20261011')/name):
                    raise ValueError('Cached probe tensor or oracle changed')
        expected = read_json(root/'expected.json')
        for host, machine in [('host', 'arm64'), ('pi', 'armv6l')]:
            path = Path(f'runs/{variant}-probe-{host}-20261011.json')
            report = read_json(path)
            inputs[str(path)] = sha256(path)
            if report['machine'] != machine or report['manifest_sha256'] != sha256(root/'manifest.json') or report['script_sha256'] != manifest['files']['probe.py']:
                raise ValueError('Runtime or bundle identity differs')
            for key in ('both_methods_simultaneous', 'inference_tested', 'camera_tested', 'motors_commanded',
                        'esp32_commanded', 'active_model_changed', 'development_evaluated', 'test_evaluated',
                        'deployment_approved', 'qualification_achieved'):
                if report[key] is not False:
                    raise ValueError('Probe exceeds declared scope')
            fast = variant == 'cached-decoder'
            parity = report['cross_host_max_box_delta' if fast else 'parity_max_box_delta']
            if any(not np.isfinite(v) or v < 0 or v > .001 for v in parity.values()):
                raise ValueError('Cross-host parity failed')
            if fast and report['native_outputs_exact'] is not True:
                raise ValueError('Exact native decoder parity failed')
            summary = recount(report, expected, FAST_ROUNDS if fast else BASE_ROUNDS)
            result = dict(raw=str(path), raw_sha256=sha256(path), machine=machine, numpy=report['numpy'],
                summary=summary, parity_max_box_delta=parity,
                health_before={k: v for k, v in report['health_before'].items() if k != 'process_snapshot'},
                health_after={k: v for k, v in report['health_after'].items() if k != 'process_snapshot'})
            if fast:
                result['native_outputs_exact'] = True
                result['effects'] = {}
                for method in ('frozen', 'neighbor'):
                    old, new = summary[method]['mean_ms'], summary['cached_'+method]['mean_ms']
                    rows = report['timings']
                    per_grid = []
                    for index in range(42):
                        a = np.mean([r['elapsed_ms'] for r in rows if r['index'] == index and r['method'] == method])
                        b = np.mean([r['elapsed_ms'] for r in rows if r['index'] == index and r['method'] == 'cached_'+method])
                        per_grid.append(float(a-b))
                    result['effects'][method] = dict(mean_saved_ms=old-new, percent_time_reduction=100*(old-new)/old,
                        speed_ratio=old/new, grids_with_lower_mean=sum(v > 0 for v in per_grid),
                        per_grid_saved_ms=per_grid)
            results[f'{variant}_{host}'] = result
    inputs[__file__] = sha256(__file__)
    write_json(OUT, dict(inputs_sha256=inputs, results=results,
        scope='Decoder-only sequential paired timing. TRAIN tensors from rejected accuracy candidates; caching changes implementation, not predictions. No model promotion or held-out accuracy evidence. Temperature and process snapshots do not prove full-system isolation.',
        inference_tested=False, camera_tested=False, deployment_approved=False, qualification_achieved=False))
    print(results['cached-decoder_pi']['effects'], flush=True)


if __name__ == '__main__':
    audit()
