"""Render the12TRAIN annotation/teacher disagreements for direct visual review."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.balloon_scene_label_audit import OUTPUT as AUDIT
from scripts.balloon_scene_constraints import scene_pixels
from scripts.prepare_grid_clutter_review import BASE
from scripts.prepare_openimages_balloon_review import sheets

OUTPUT = Path('data/scene-teacher-review-20261011')


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = read_json(AUDIT)
    if sha256(audit['detail']['path']) != audit['detail']['sha256']:
        raise ValueError('Annotation audit changed')
    pixels, identities = scene_pixels()
    rows = read_json(audit['detail']['path'])
    paths = {f['source']: BASE/f['path'] for f in read_json(BASE/'manifest.json')['frames']}
    chosen = [i for i, r in enumerate(rows) if
        (r['original']['category'] == 'empty_crop' and r['raw_accepted']) or
        (r['original']['category'].startswith('matched_') and not r['raw_accepted'])]
    if len(chosen) != 12:
        raise ValueError('Disagreement set changed')
    OUTPUT.mkdir(parents=True)
    samples = []
    for i in chosen:
        row, identity = rows[i], identities[i]
        path = paths[row['source']]
        if sha256(path) != identity['frame_sha256']:
            raise ValueError('Source pixels changed')
        a, b, c, d = row['crop_box']
        with Image.open(path) as im:
            crop = im.convert('RGB').crop((a, b, c, d))
        np.testing.assert_array_equal(np.asarray(crop.resize((64, 64), Image.Resampling.BILINEAR), np.float32), pixels[i])
        output = OUTPUT/f'{i:03d}.png'
        crop.save(output)
        samples.append(dict(id=i, frame_id=row['source'], path=output.name, sha256=sha256(output),
                            source_path=str(path), **row))
    sheets(OUTPUT, 'crops', samples, 3, (240, 190))
    write_json(OUTPUT/'review.json', dict(samples=samples, audit_sha256=sha256(AUDIT),
        script_sha256=sha256(__file__), labels_approved=False, scope='TRAIN-only review queue'))
    print(dict(path=str(OUTPUT), queue_sha256=sha256(OUTPUT/'review.json')))


if __name__ == '__main__':
    prepare()
