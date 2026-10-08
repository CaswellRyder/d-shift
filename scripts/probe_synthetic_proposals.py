"""Read-only synthetic proposal coverage probe; no model or Pi speed claims."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.tracking import iou
from dtr.vision import proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text())
    path = args.annotations.parent / data['image']
    image = np.asarray(Image.open(path).convert('RGB'))
    assert image.shape[:2] == (data['height'], data['width'])
    results = []
    for width, height in ((320, 240), (640, 480)):
        rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        for task, profile in (('goal', 'balloon_components'), ('goal', 'orange_v3'),
                              ('goal', 'goal_edges'), ('goal', 'orange_local'),
                              ('balloon', 'balloon_components')):
            if profile == 'goal_edges' and width > 320:
                results.append(dict(size=[width, height], task=task, profile=profile,
                                    skipped='Edge-search implementation is bounded to 320x240'))
                continue
            candidates = proposals(rgb, task, limit=12, profile=profile)
            coverage = []
            for obj in data['objects']:
                if (obj['label'].endswith('_balloon')) != (task == 'balloon'):
                    continue
                box = [value * (width / data['width'] if i % 2 == 0
                                else height / data['height'])
                       for i, value in enumerate(obj['box'])]
                overlap = max((iou(box, candidate['box']) for candidate in candidates),
                              default=0)
                coverage.append(dict(label=obj['label'], box=box,
                                     best_iou=float(overlap), covered=overlap >= .5))
            results.append(dict(size=[width, height], task=task, profile=profile,
                                candidate_count=len(candidates), coverage=coverage))
    rendered = json.dumps(dict(synthetic=True, classification_measured=False,
                          pi_timing_measured=False,
                          image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          results=results), indent=2)
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
