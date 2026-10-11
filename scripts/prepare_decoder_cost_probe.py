"""Package exact decoder sources and a predetermined uniform TRAIN tensor sample."""
from pathlib import Path
import shutil
import tarfile

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_scale_balanced_search import checked_panels
from scripts.balloon_dense_runtime import prepare_input
from scripts.research_blue_peak_blend import decode as frozen
from scripts.independent_neighbor_decoder import decode as neighbor

AUDIT = Path('research/balloon-red-blue-20261008/center-presence-localizer-20261011.json')
AUDIT_SHA = '6f74716d8060306921acc960f199d1515ce51fd9e0f97eb056f51f7df73dbbdd'
ROOT = Path('artifacts/decoder-cost-probe-20261011')
ARCHIVE = Path('artifacts/decoder-cost-probe-20261011.tar.gz')


def prepare():
    if ROOT.exists() or ARCHIVE.exists():
        raise FileExistsError('Preserve probe bundle')
    if sha256(AUDIT) != AUDIT_SHA:
        raise ValueError('Audited model source changed')
    audit = read_json(AUDIT)
    if any(sha256(p) != h for p, h in audit['inputs_sha256'].items()):
        raise ValueError('Audited source dependency changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    records, pixels = [], []
    for panel, rows in checked_panels().items():
        # Fixed uniform positions, independent of confidence or decoding outcome.
        for index in range(0, len(rows), 4):
            frame, rgb = rows[index]
            records.append(dict(panel=panel, index=index, source=frame['source'], frame_sha256=frame['frame_sha256']))
            pixels.append(prepare_input(rgb)[0])
    if len(records) != 21:
        raise ValueError('Unexpected source sampling coverage')
    outputs, identities = [], []
    models = {}
    for seed in (47, 48):
        record = audit['training'][f'presence{seed}']
        for filename, digest in record['artifacts_sha256'].items():
            if sha256(Path(record['run'])/filename) != digest:
                raise ValueError('Audited model artifact changed')
        path = Path(record['run'])/'proposal.keras'
        model = keras.models.load_model(path, compile=False)
        values = model(np.stack(pixels), training=False).numpy()
        outputs.extend(values)
        identities.extend(dict(r, seed=seed) for r in records)
        models[str(seed)] = dict(path=str(path), sha256=sha256(path))
    values = np.stack(outputs)
    ROOT.mkdir(parents=True)
    np.save(ROOT/'outputs.npy', values, allow_pickle=False)
    write_json(ROOT/'expected.json', dict(frozen=[frozen(v, True) for v in values],
        independent_neighbor=[neighbor(v) for v in values]))
    copies = {'scripts/pi_decoder_cost_probe.py': 'probe.py',
        'scripts/research_blue_peak_blend.py': 'scripts/research_blue_peak_blend.py',
        'scripts/independent_neighbor_decoder.py': 'scripts/independent_neighbor_decoder.py',
        'src/dtr/tracking.py': 'dtr/tracking.py'}
    for source, name in copies.items():
        (ROOT/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, ROOT/name)
    files = [*copies.values(), 'outputs.npy', 'expected.json']
    write_json(ROOT/'manifest.json', dict(files={name: sha256(ROOT/name) for name in files},
        audit_sha256=AUDIT_SHA, script_sha256=sha256(__file__), source_files={p: sha256(p) for p in copies},
        models=models, frames=identities, rounds=['frozen', 'independent_neighbor', 'independent_neighbor', 'frozen'],
        scope='Every fourth source in each82-frame TRAIN panel,21photos x two seed outputs. No images or heldout data transferred. No inference or camera benchmark. Declared after accuracy rejection to measure decoder cost, not to promote the rejected models.',
        deployment_approved=False, qualification_achieved=False))
    with tarfile.open(ARCHIVE, 'w:gz') as archive:
        for name in [*files, 'manifest.json']:
            archive.add(ROOT/name, arcname=name)
    print(dict(root=str(ROOT), archive=str(ARCHIVE), archive_sha256=sha256(ARCHIVE)), flush=True)


if __name__ == '__main__':
    prepare()
