"""Verify native output against host tensors and exact frozen selection decisions."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.pi_dense_compatibility_probe import verified
from scripts.audit_balloon_dense_proposals import decode
from scripts.research_dense_supported_selection import supported_selection


def verify(root, results):
    root, results = Path(root), Path(results)
    manifest = verified(root)
    report = read_json(results / 'report.json')
    if (report['manifest_sha256'] != sha256(root / 'manifest.json')
            or manifest['policy_sha256'] != sha256('scripts/research_dense_supported_selection.py')
            or manifest['decoder_sha256'] != sha256('scripts/audit_balloon_dense_proposals.py')
            or set(report['models']) != set(manifest['models'])):
        raise ValueError('Report or policy identity changed')
    rows, evidence = read_json(root / 'golden.json'), {}
    if not rows or len(rows) != manifest['frames']:
        raise ValueError('Incomplete golden frames')
    for seed, filename in manifest['models'].items():
        item = report['models'][seed]
        path = results / f'predictions-{seed}.npy'
        if (item['model_sha256'] != sha256(root / filename)
                or item['output_sha256'] != sha256(path) or item['calls'] != len(rows)):
            raise ValueError('Model or tensor identity changed')
        expected = np.load(root / f'host-{seed}.npy', allow_pickle=False)
        actual = np.load(path, allow_pickle=False)
        if (actual.shape != expected.shape or actual.shape != (len(rows),15,20,6)
                or actual.dtype != np.float32 or not np.isfinite(actual).all()):
            raise ValueError('Invalid tensor outputs')
        delta = float(np.max(np.abs(actual-expected)))
        if delta > .002:
            raise ValueError('Native tensor tolerance exceeded')
        for row, value in zip(rows, actual, strict=True):
            if supported_selection(row['raw'], decode(value)) != row['selected'][seed]:
                raise ValueError('Native output changed selection decisions')
        evidence[seed] = dict(frames=len(rows), max_tensor_delta=delta, exact_selection_parity=True,
            model_sha256=item['model_sha256'], output_sha256=item['output_sha256'],
            runtime_version=item.get('runtime_version'))
    return dict(models=evidence, machine=report['machine'],
        manifest_sha256=sha256(root / 'manifest.json'), report_sha256=sha256(results / 'report.json'),
        scope='Native tensor compatibility; proposal search, classifier and decoder/selection were NOT executed on Pi by this probe.',
        camera_tested=False, isolated_speed_measured=False, deployment_approved=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','results','output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = verify(a.base, a.results)
    write_json(a.output, result)
    print(result)
