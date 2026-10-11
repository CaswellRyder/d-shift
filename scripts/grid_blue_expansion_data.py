"""Sparse proposal supervision from reviewed blue balloons and real blue clutter.

Reuse existing TRAIN labels, never development images. Preserve the original
grouped holdout and do not infer whole-image background from sparse annotations.
"""
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from dtr.data import sha256
from scripts.blue_false_alarm_data import checked_extra as checked_negatives
from scripts.openimages_blue_expansion import checked_extra as checked_positives
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.balloon_dense_public_data import encode_public
from scripts.train_balloon_dense_proposals import checked_data

BASE = Path('data/balloon-dense-public-admit-20261011')
BASE_SHA = '029b91b1849c3cc9bda1b5e7b21d5d809e61e2142c476ee030c9ae6de1fc773b'


def combine(positives, negatives):
    grouped = defaultdict(lambda: dict(truth=[], backgrounds=[], positive_ids=[], negative_ids=[]))
    for row in positives:
        if row['label'] != 'blue_balloon' or row['split'] != 'train':
            raise ValueError('Require reviewed TRAIN blue boxes')
        group = grouped[row['source_image']]
        group['truth'].append(dict(label='blue_balloon', box=row['box']))
        group['positive_ids'].append(row['id'])
    for row in negatives:
        if row['label'] != 'background' or row['split'] != 'train':
            raise ValueError('Require reviewed TRAIN background crops')
        group = grouped[row['source_image']]
        group['backgrounds'].append(row['crop_box'])
        group['negative_ids'].append(row['id'])
    return dict(grouped)


def expected(mode):
    if mode not in ('control', 'admit'):
        raise ValueError('Unknown paired arm')
    if sha256(BASE / 'manifest.json') != BASE_SHA:
        raise ValueError('Original proposal data changed')
    doc, x, y, folds, groups = checked_data(BASE)
    selected = combine(checked_positives(), checked_negatives())
    sources = {row['source']: (row, rgb) for row, rgb in source_frames()}
    extra, omitted = [], []
    hold = [row for row, fold in zip(doc['frames'], folds, strict=True) if fold == 0]
    held_sources = {r['source'] for r in hold}
    held_hashes = {r['source_sha256'] for r in hold}
    for source, annotations in sorted(selected.items()):
        origin, rgb = sources[source]
        if source in held_sources or origin['source_sha256'] in held_hashes:
            raise ValueError('New supervision crosses original grouped holdout')
        target, backgrounds, skipped, collisions = encode_public(annotations['truth'], annotations['backgrounds'])
        # Small negative crops can cover no complete16px cell. Keep that
        # limitation visible; do not expand the reviewed negative geometry.
        record = dict(source=source, source_sha256=origin['source_sha256'],
            source_path=origin['source_path'], source_file_sha256=origin['source_file_sha256'],
            truth=annotations['truth'], negative_regions=backgrounds,
            positive_ids=annotations['positive_ids'], negative_ids=annotations['negative_ids'],
            skipped_negative_regions=skipped, box_collisions=collisions,
            supervised_heat_cells=int((target[..., 2:4] > 0).sum()),
            box_cells=int(target[..., 8].sum()))
        if not record['supervised_heat_cells']:
            omitted.append(record)
            continue
        extra.append((record, cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32), target))
    if not extra:
        raise ValueError('No new usable sparse supervision')
    fit_indices = np.flatnonzero(folds != 0)
    choices = np.random.default_rng(47).choice(fit_indices, len(extra), replace=False)
    images, targets = list(x), list(y)
    rows, output_folds, output_groups = list(doc['frames']), list(folds), list(groups)
    for (record, pixels, target), index in zip(extra, choices, strict=True):
        if mode == 'admit':
            rows.append(dict(record, expansion_kind='reviewed_blue_regions'))
            images.append(pixels)
            targets.append(target)
            output_folds.append(1)
            output_groups.append(record['source'])
        else:
            rows.append(dict(doc['frames'][int(index)], expansion_kind='resampled_fit', base_index=int(index)))
            images.append(x[index])
            targets.append(y[index])
            output_folds.append(int(folds[index]))
            output_groups.append(groups[index])
    stats = dict(base_frames=len(x), extra_frames=len(extra), proposed_sources=len(selected),
        omitted_zero_cell_sources=omitted, control_indices=choices.tolist(),
        extra_blue_boxes=sum(len(r['truth']) for r, _, _ in extra),
        extra_box_cells=sum(r['box_cells'] for r, _, _ in extra),
        extra_heat_cells=sum(r['supervised_heat_cells'] for r, _, _ in extra),
        extra_source_kinds=dict(Counter(sources[r['source']][0]['origin'] for r, _, _ in extra)),
        extra_reviewed_records=[r for r, _, _ in extra])
    return dict(mode=mode, records=rows, images=np.asarray(images, np.float32),
        targets=np.asarray(targets, np.float32), folds=np.asarray(output_folds), groups=output_groups,
        stats=stats, base_manifest_sha256=BASE_SHA,
        source_sha256={p: sha256(p) for p in (
            __file__, 'configs/openimages-blue-expansion2-20261011.json',
            'configs/blue-false-alarm-review-20261011.json',
            'scripts/blue_false_alarm_data.py', 'scripts/openimages_blue_expansion.py',
            'scripts/prepare_blue_false_alarm_review.py', 'scripts/balloon_dense_public_data.py',
            'scripts/train_balloon_dense_proposals.py', 'scripts/prepare_balloon_dense_proposals.py')})


if __name__ == '__main__':
    # A read-only data audit; does not create training artifacts or approve a run.
    for arm in ('control', 'admit'):
        data = expected(arm)
        print(dict(mode=arm, images=list(data['images'].shape), targets=list(data['targets'].shape),
            fit=int((data['folds'] != 0).sum()), hold=int((data['folds'] == 0).sum()),
            stats={k: v for k, v in data['stats'].items() if k not in ('extra_reviewed_records', 'omitted_zero_cell_sources')}))
