"""Reuse the exact frozen TRAIN tensor bundle; do not run or retrain any model."""
from pathlib import Path
import shutil
import tarfile

from dtr.data import read_json, sha256, write_json

SOURCE = Path('artifacts/decoder-cost-probe-20261011')
ROOT = Path('artifacts/cached-decoder-probe-20261011')
ARCHIVE = Path('artifacts/cached-decoder-probe-20261011.tar.gz')


def prepare():
    if ROOT.exists() or ARCHIVE.exists():
        raise FileExistsError('Preserve probe bundle')
    if sha256(str(SOURCE)+'.tar.gz') != 'f73a023791e1903dadb5512467fd5ca6f25c9ea58f4ae096135b9f34d13da473':
        raise ValueError('Original probe archive changed')
    original = read_json(SOURCE/'manifest.json')
    for name, digest in original['files'].items():
        if sha256(SOURCE/name) != digest:
            raise ValueError('Original probe identity changed')
    copies = {'scripts/pi_cached_decoder_probe.py': 'probe.py',
        'scripts/pi_decoder_cost_probe.py': 'scripts/pi_decoder_cost_probe.py',
        'scripts/cached_balloon_decoder.py': 'scripts/cached_balloon_decoder.py',
        'scripts/research_blue_peak_blend.py': 'scripts/research_blue_peak_blend.py',
        'scripts/independent_neighbor_decoder.py': 'scripts/independent_neighbor_decoder.py',
        'src/dtr/tracking.py': 'dtr/tracking.py'}
    for path, digest in original['source_files'].items():
        if sha256(path) != digest:
            raise ValueError('Reference decoder sources changed')
    ROOT.mkdir(parents=True)
    for source, name in copies.items():
        (ROOT/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, ROOT/name)
    for name in ('outputs.npy', 'expected.json'):
        shutil.copy2(SOURCE/name, ROOT/name)
    files = [*copies.values(), 'outputs.npy', 'expected.json']
    write_json(ROOT/'manifest.json', dict(files={name: sha256(ROOT/name) for name in files},
        source_manifest_sha256=sha256(SOURCE/'manifest.json'), script_sha256=sha256(__file__),
        source_files={p: sha256(p) for p in copies}, frames=original['frames'], models=original['models'],
        scope='Exact same tensors and expected outputs as the frozen decoder-cost probe. No new data or model selection.',
        deployment_approved=False, qualification_achieved=False))
    with tarfile.open(ARCHIVE, 'w:gz') as archive:
        for name in [*files, 'manifest.json']:
            archive.add(ROOT/name, arcname=name)
    print(dict(archive=str(ARCHIVE), sha256=sha256(ARCHIVE)), flush=True)


if __name__ == '__main__':
    prepare()
