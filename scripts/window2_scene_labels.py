"""Validated, manually completed TRAIN scene labels; no model-derived truth."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dtr.data import read_json, sha256, write_json
from scripts.prepare_openimages_balloon_review import checked_path

REVIEW = Path('configs/blue-balloon-window2-scenes-20261011.json')
LABELS = {'red_balloon', 'blue_balloon'}


def validate_partition(review, queue):
    ids = {r['id'] for r in queue['samples']}
    positive = set(map(int, review['labels']))
    excluded = set(map(int, review['excluded_annotations']))
    if positive & excluded or positive | excluded != ids:
        raise ValueError('Every source annotation requires one explicit disposition')
    if len(ids) != len(queue['samples']):
        raise ValueError('Duplicate source annotation IDs')
    frames = {f['id'] for f in queue['frames']}
    if set(review['complete_frame_ids']) != frames or len(review['complete_frame_ids']) != len(frames):
        raise ValueError('Require all twelve reviewed TRAIN frames without duplicates')
    manual_ids = [a['id'] for a in review['manual_annotations']]
    if len(set(manual_ids)) != len(manual_ids) or set(manual_ids) & ids:
        raise ValueError('Manual annotation IDs must be unique and separate from source')
    if not set(review['labels'].values()) <= LABELS:
        raise ValueError('Invalid target label')
    samples = {r['id']: r for r in queue['samples']}
    for i in positive:
        r = samples[i]
        if r['split'] != 'train' or int(r['source_annotation'].get('IsGroupOf', 0)):
            raise ValueError('Source group or non-TRAIN annotation cannot be an individual target')
    for a in review['manual_annotations']:
        if a['frame_id'] not in frames or a['label'] not in LABELS or not a['reason']:
            raise ValueError('Invalid manual target')
        if 'replaces_group' in a:
            i = a['replaces_group']
            if i not in excluded or samples[i]['frame_id'] != a['frame_id'] or not int(samples[i]['source_annotation'].get('IsGroupOf', 0)):
                raise ValueError('Manual group replacement must reference an excluded same-frame group')


def scaled_box(box, width, height):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Invalid annotation coordinates')
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Annotation outside source image')
    return (values * np.asarray([320 / width, 240 / height] * 2)).tolist()


def checked_frames(require_approved=True):
    review = read_json(REVIEW)
    if require_approved and review['scene_evaluation_approved'] is not True:
        raise ValueError('Corrected scene labels await visual approval')
    if any(review[k] is not False for k in ('training_approved', 'development_evaluated', 'test_evaluated', 'deployment_approved')):
        raise ValueError('Only TRAIN scene diagnosis authorized')
    root = Path(review['queue'])
    if sha256(root / 'review.json') != review['queue_sha256']:
        raise ValueError('Source review queue changed')
    q = read_json(root / 'review.json')
    validate_partition(review, q)
    result = []
    for f in q['frames']:
        with Image.open(checked_path(Path(review['original_root']), f['original_path'], f['original_sha256'])) as im:
            width, height = im.size
        with Image.open(checked_path(root, f['path'], f['sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        if rgb.shape != (240, 320, 3):
            raise ValueError('Changed scan dimensions')
        truth = []
        for r in q['samples']:
            if r['frame_id'] == f['id'] and str(r['id']) in review['labels']:
                truth.append(dict(annotation_id=r['id'], label=review['labels'][str(r['id'])],
                                  box=scaled_box(r['box'], 320, 240), origin='source_individual'))
        for a in review['manual_annotations']:
            if a['frame_id'] == f['id']:
                truth.append(dict(annotation_id=a['id'], label=a['label'],
                                  box=scaled_box(a['box_original'], width, height), origin='manual_source_review'))
        result.append((dict(id=f['id'], source='openimages/train/' + f['image_id'],
                            frame_sha256=f['sha256'], truth=truth), rgb))
    return result


def render():
    output = Path('data/window2-corrected-scene-overlays-20261011')
    if output.exists():
        raise FileExistsError(output)
    frames = checked_frames(require_approved=False)
    output.mkdir(parents=True)
    sheet = Image.new('RGB', (960, 1040), 'white')
    for index, (f, rgb) in enumerate(frames):
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        for t in f['truth']:
            color = 'red' if t['label'] == 'red_balloon' else 'cyan'
            draw.rectangle(t['box'], outline=color, width=1)
            draw.text(t['box'][:2], str(t['annotation_id']), fill='black', stroke_fill='white', stroke_width=1)
        im.save(output / f"{f['id']}.png")
        x, y = index % 3 * 320, index // 3 * 260
        sheet.paste(im, (x, y + 20))
        ImageDraw.Draw(sheet).text((x, y), str(f['id']), fill='black')
    sheet.save(output / 'sheet.png')
    write_json(output / 'receipt.json', dict(review_sha256=sha256(REVIEW), script_sha256=sha256(__file__),
        images_sha256={p.name: sha256(p) for p in output.glob('*.png')},
        scope='Manual source-label overlay review only, before any model prediction.'))


if __name__ == '__main__':
    render()
