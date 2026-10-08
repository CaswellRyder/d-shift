"""Build visibly reviewed synthetic goal crops; never mix with real evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.runtime import Predictor
from dtr.vision import proposals, observe


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    doc = json.loads(args.annotations.read_text())
    if doc['synthetic'] is not True or doc['split'] != 'train':
        raise ValueError('Require synthetic train-only inputs')
    if args.output.exists():
        raise FileExistsError(args.output)
    predictor = Predictor(args.model, allow_unvalidated=True)
    if predictor.metadata['task'] != 'goal':
        raise ValueError('Goal model required')
    args.output.mkdir(parents=True)
    samples, diagnostics = [], []
    for scene in doc['scenes']:
        path = args.annotations.parent/scene['image']
        with Image.open(path) as opened:
            if opened.size != (scene['width'], scene['height']):
                raise ValueError('Image size mismatch')
            rgb = cv2.resize(np.asarray(opened.convert('RGB')), (320, 240))
        crops = []
        for obj in scene['objects']:
            if obj['label'] not in predictor.metadata['classes']:
                raise ValueError('Unknown goal label')
            a, b, c, d = [int(v*(320/scene['width'] if i % 2 == 0 else 240/scene['height']))
                           for i, v in enumerate(obj['box'])]
            pad = int(max(c-a, d-b)*.12)
            crops.append((obj['label'], [max(0, a-pad), max(0, b-pad),
                                        min(320, c+pad), min(240, d+pad)]))
        if scene['goal_absent']:
            if scene['objects']:
                raise ValueError('Goal-absent scene has positive labels')
            # Labels come from full-scene visual review, NOT model predictions.
            candidates = proposals(rgb, 'goal', limit=12, profile='balloon_components')
            crops.extend(('background', c['crop_box']) for c in candidates[:6])
        diagnostics.append(dict(image=scene['image'], image_sha256=digest(path),
                                baseline=observe(rgb, predictor, profile='balloon_components'),
                                gap9=observe(rgb, predictor, profile='goal_gap9')))
        for label, box in crops:
            a, b, c, d = box
            if c <= a or d <= b:
                raise ValueError('Empty crop')
            crop = rgb[b:d, a:c]
            relative = f'{len(samples):03d}.png'
            Image.fromarray(crop).save(args.output/relative)
            samples.append(dict(path=relative, label=label, split='train',
                                synthetic=True, session=scene['source_group'],
                                source_image=scene['image'], source_sha256=digest(path),
                                sha256=digest(args.output/relative), crop_box=box,
                                baseline_prediction=predictor.predict(crop)))
    manifest = dict(synthetic=True, task='goal', split='train',
                    classes=predictor.metadata['classes'], annotation_sha256=digest(args.annotations),
                    model_sha256=digest(args.model), script_sha256=digest(__file__),
                    label_origin=doc['annotation_origin'], evaluation_eligible=False,
                    deployment_approved=False, samples=samples)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print([(r['label'], r['baseline_prediction']['label'],
            round(r['baseline_prediction']['score'], 3)) for r in samples])


if __name__ == '__main__':
    main()
