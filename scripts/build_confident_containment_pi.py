"""Immutable 205-scene research replay, including TRAIN behavior changes."""
import inspect
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts import compare_balloon_containment as containment
from scripts.evaluate_current_confident_containment import OUTPUT as SCENES, RAW, checked_reference
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.pi_red_blue_bench import verify_bundle
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_confident_containment import select
from scripts.replay_balloon_suppression import raw

SOURCE = Path('runs/blue-peak-blend-pi-20261011')
OUTPUT = Path('runs/current-containment-pi-20261011')


def frozen_source():
    return ('"""Exact extracted containment functions; source hashes in bundle manifest."""\n'
        'import math\nLABELS = ("red_balloon", "blue_balloon")\n\n'
        + '\n\n'.join(inspect.getsource(getattr(containment, name))
                       for name in ('area', 'nested', 'before_parts', 'confidence_ordered')))


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference, previous = checked_reference()
    evidence = read_json(SCENES)
    if (not evidence['eligible_for_pi_comparison'] or evidence['raw']['sha256'] != sha256(RAW)
            or any(sha256(p) != h for p, h in evidence['helpers_sha256'].items())
            or evidence['script_sha256'] != sha256('scripts/evaluate_current_confident_containment.py')
            or evidence['protocol_sha256'] != sha256('configs/current-confidence-containment-20261011.json')):
        raise ValueError('Host replay evidence changed')
    revised = read_json(RAW)
    source = verify_bundle(SOURCE)
    if (source['classifier_sha256'] != reference['classifier_sha256']
            or source['head_localizer_sha256'] != reference['models']['background']['sha256']):
        raise ValueError('Bundle model identity changed')
    for relative in source['files']:
        target = OUTPUT/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/relative, target)
    train = {s['source']: s for s, _ in source_frames() if s['origin'] == 'complete_fit'}
    indexed = {(r['panel'], r['id']): r for r in read_json(SOURCE/'decoder-frames.json')}
    extracted = {}
    code = frozen_source()
    exec(compile(code, 'current_containment_frozen.py', 'exec'), extracted)
    frames = []
    for panel, variant in revised.items():
        expected_pixels = None
        if panel != 'training':
            reviewed, _ = reviewed_frames('data/'+panel, 'configs/'+panel+'.json')
            expected_pixels = {r['id']: (r, rgb) for r, rgb in reviewed}
        for old, new in zip(previous[panel]['background/blend']['frames'], variant['frames'], strict=True):
            if any(old[k] != new[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                raise ValueError('Different host scene identity')
            if select(raw(old['detections'])) != new['detections']:
                raise ValueError('Host selection changed')
            if extracted['confidence_ordered'](old['detections']) != new['detections']:
                raise ValueError('Extracted Pi policy differs')
            if panel == 'training':
                item = train[old['source']]
                if item['source_file_sha256'] != old['frame_sha256'] or item['truth'] != old['truth']:
                    raise ValueError('TRAIN pixel identity changed')
                relative = f"training/{old['id']:03d}.png"
                (OUTPUT/'training').mkdir(exist_ok=True)
                shutil.copyfile(item['source_path'], OUTPUT/relative)
            else:
                item = indexed[(panel, old['id'])]
                if any(item[k] != old[k] for k in ('source', 'truth', 'frame_sha256')):
                    raise ValueError('Development pixel identity changed')
                relative = item['path']
                record, pixels = expected_pixels[old['id']]
                with Image.open(OUTPUT/relative) as image:
                    copied = np.asarray(image.convert('RGB'))
                if (record['sha256'] != old['frame_sha256'] or record['provisional_truth'] != old['truth']
                        or not np.array_equal(copied, pixels)):
                    raise ValueError('Prepared replay pixels/orientation differ from reviewed scene')
            # Source image hashes and prepared/rotated replay PNG hashes are distinct.
            expected_hash = old['frame_sha256'] if panel == 'training' else source['files'][relative]
            if sha256(OUTPUT/relative) != expected_hash:
                raise ValueError('Copied frame hash differs')
            frames.append(dict({k: old[k] for k in ('id', 'source', 'truth', 'frame_sha256')},
                panel=panel, path=relative, replay_file_sha256=expected_hash,
                expected=dict(parent=old['detections'], confidence=new['detections'])))
    if len(frames) != 205:
        raise ValueError('Incomplete scene set')
    (OUTPUT/'scripts/current_containment_frozen.py').write_text(code)
    shutil.copyfile('scripts/pi_confident_containment.py', OUTPUT/'scripts/pi_confident_containment.py')
    write_json(OUTPUT/'containment-frames.json', frames)
    write_json(OUTPUT/'bundle.json', dict(
        files={str(p.relative_to(OUTPUT)): sha256(p) for p in OUTPUT.rglob('*') if p.is_file()},
        source_bundle_sha256=sha256(SOURCE/'bundle.json'), host_scenes_sha256=sha256(SCENES),
        host_raw_sha256=sha256(RAW), builder_sha256=sha256(__file__),
        extracted_source_sha256=sha256('scripts/compare_balloon_containment.py'),
        classifier_sha256=source['classifier_sha256'], head_localizer_sha256=source['head_localizer_sha256'],
        order=['parent', 'confidence', 'confidence', 'parent'],
        scope='Isolated ABBA full-pipeline replay,205unique TRAIN/development rows per block; no camera/actuation/deployment.',
        deployment_approved=False, qualification_achieved=False))
    print(dict(root=str(OUTPUT), frames=len(frames), bundle_sha256=sha256(OUTPUT/'bundle.json')), flush=True)


if __name__ == '__main__':
    build()
