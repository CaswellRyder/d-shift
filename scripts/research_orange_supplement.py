"""Train-only orange proposal experiment; never modifies deployment defaults.

Preserve baseline geometry and append at most two nonduplicate orange candidates.
This increases the proposal cap from 12 to 14, NOT the neural per-frame budget.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from dtr.tracking import iou
from dtr.vision import proposals


def supplement(baseline, local, extra=2):
    if not isinstance(extra, int) or isinstance(extra, bool) or not 0 <= extra <= 2:
        raise ValueError('Supplement must be an integer from 0 to 2')
    selected = list(baseline)
    added = 0
    for candidate in local:
        if added >= extra:
            break
        if candidate['color_group'] != 0:
            continue
        if any(iou(candidate['box'], old['box']) > .65 for old in selected):
            continue
        selected.append(candidate)
        added += 1
    return selected


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, output, source='orange_local'):
    if root.name != 'train':
        raise ValueError('This tuning experiment requires the train split')
    if output.exists():
        raise FileExistsError(output)
    annotations = root / '_annotations.coco.json'
    doc = json.loads(annotations.read_text())
    names = {row['id']: row['name'] for row in doc['categories']}
    labels = defaultdict(list)
    for row in doc['annotations']:
        labels[row['image_id']].append(row)
    counts, sizes = {}, {}
    timings = defaultdict(list)
    frames = []
    for info in sorted(doc['images'], key=lambda r: r['file_name'])[::8]:
        path = root / info['file_name']
        with Image.open(path) as opened:
            rgb = cv2.resize(np.asarray(opened.convert('RGB')), (320, 240))
        start = time.perf_counter()
        baseline = proposals(rgb, 'goal', limit=12, profile='balloon_components')
        middle = time.perf_counter()
        local = proposals(rgb, 'goal', limit=12 if source == 'orange_local' else 64,
                          profile=source)
        trial = supplement(baseline, local)
        end = time.perf_counter()
        timings['baseline_ms'].append((middle-start)*1000)
        timings['supplement_total_ms'].append((end-start)*1000)
        frame = dict(file=info['file_name'], sha256=digest(path),
                     baseline_candidates=len(baseline), trial_candidates=len(trial), targets=[])
        for ann in labels[info['id']]:
            label = names[ann['category_id']]
            if not label.endswith('Goal') or not label.startswith(('Orange', 'Yellow')):
                continue
            x, y, w, h = ann['bbox']
            box = [x*320/info['width'], y*240/info['height'],
                   (x+w)*320/info['width'], (y+h)*240/info['height']]
            hit_base = max((iou(box, r['box']) for r in baseline), default=0) >= .5
            hit_trial = max((iou(box, r['box']) for r in trial), default=0) >= .5
            longest = max(box[2]-box[0], box[3]-box[1])
            size = 'under16' if longest < 16 else '16to32' if longest < 32 else '32plus'
            for group, key in ((counts, label), (sizes, label+'/'+size)):
                value = group.setdefault(key, dict(targets=0, baseline=0, trial=0))
                value['targets'] += 1
                value['baseline'] += hit_base
                value['trial'] += hit_trial
            frame['targets'].append(dict(label=label, box=box, baseline=hit_base, trial=hit_trial))
        frames.append(frame)
    report = dict(split='train', test_evaluated=False, deployment_approved=False,
                  scope='proposal coverage only; not classifier precision or Pi timing',
                  image_size=[320, 240], size_definition='longest box side',
                  baseline_cap=12, trial_cap=14, neural_inference_measured=False,
                  supplemental_source=source,
                  annotation_sha256=digest(annotations), script_sha256=digest(__file__),
                  vision_sha256=digest(Path(__file__).parents[1]/'src/dtr/vision.py'),
                  counts=counts, size_counts=sizes, frames=frames,
                  local_mac_ms={k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)))
                                for k, v in timings.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path,
                        default=Path('data/roboflow-dtr-v10-grouped/coco/train'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', choices=['orange_local', 'balloon_components', 'goal_gap9'],
                        default='orange_local')
    args = parser.parse_args()
    audit(args.train_root, args.output, args.source)
