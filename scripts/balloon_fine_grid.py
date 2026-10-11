"""Research grid generalization; frozen deployed and prior experiment code stays untouched."""
from pathlib import Path

import cv2
import numpy as np

from dtr.data import read_json, sha256
from dtr.tracking import iou

PROTOCOL = Path('configs/balloon-fine-grid-trial-20261011.json')


def encode_grid(truth, negatives, grid=(30, 40), complete=False):
    if grid not in ((15, 20), (30, 40)):
        raise ValueError('Unsupported grid')
    h, w = grid
    step = 320 / w
    xx, yy = np.meshgrid((np.arange(w) + .5) * step, (np.arange(h) + .5) * step)
    heat = np.zeros((h, w, 2), np.float32)
    mask = np.zeros_like(heat)
    box = np.zeros((h, w, 4), np.float32)
    box_mask = np.zeros((h, w, 1), np.float32)
    occupied, collisions = set(), set()
    for target in truth:
        a, b, c, d = target['box']
        if target['label'] not in ('red_balloon', 'blue_balloon') or not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
            raise ValueError('Invalid reviewed target')
        group = 0 if target['label'] == 'red_balloon' else 1
        cx, cy = (a + c) / 2, (b + d) / 2
        gx, gy = min(w - 1, int(cx / step)), min(h - 1, int(cy / step))
        inside = (xx >= a) & (xx < c) & (yy >= b) & (yy < d)
        inside[gy, gx] = True
        peak = np.exp(-.5 * (((xx - cx) / max((c - a) / 6, step / 2)) ** 2
                            + ((yy - cy) / max((d - b) / 6, step / 2)) ** 2))
        peak[gy, gx] = 1.
        heat[:, :, group] = np.maximum(heat[:, :, group], np.where(inside, peak, 0))
        mask[inside] = 1.
        if (gy, gx) in occupied:
            collisions.add((gy, gx))
        occupied.add((gy, gx))
        box[gy, gx] = [cx / step - gx, cy / step - gy, (c - a) / 320, (d - b) / 240]
        box_mask[gy, gx] = 1.
    for a, b, c, d in negatives:
        if not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
            raise ValueError('Invalid explicit background region')
        inside = (xx - step / 2 >= a) & (xx + step / 2 <= c) & (yy - step / 2 >= b) & (yy + step / 2 <= d)
        if np.any(mask[inside]):
            raise ValueError('Explicit background overlaps supervised balloon cell')
        mask[inside] = 1.
    for gy, gx in collisions:
        box_mask[gy, gx] = 0.
    if complete:
        background = np.ones(grid, bool)
        for row in truth:
            a, b, c, d = row['box']
            intersects = ((xx + step / 2 > a - 16) & (xx - step / 2 < c + 16)
                          & (yy + step / 2 > b - 16) & (yy - step / 2 < d + 16))
            background &= ~intersects
        mask[background] = 1.
    return np.concatenate((heat, mask, box, box_mask), axis=-1), len(collisions)


def encode_fine_safe(truth, negatives, complete=False):
    accepted, skipped = [], []
    for region in negatives:
        try:
            encode_grid(truth, accepted + [region], complete=complete)
        except ValueError as error:
            if str(error) != 'Explicit background overlaps supervised balloon cell':
                raise
            skipped.append(region)
        else:
            accepted.append(region)
    target, _ = encode_grid(truth, accepted, complete=complete)
    return target, skipped


def checked_grid_data():
    from scripts.train_balloon_dense_proposals import checked_data
    from scripts.balloon_dense_complete import validate_derivative
    protocol = read_json(PROTOCOL)
    root, base_root = Path(protocol['data']), Path(protocol['base_data'])
    if (sha256(root / 'manifest.json') != protocol['manifest_sha256']
            or sha256(base_root / 'manifest.json') != protocol['base_manifest_sha256']):
        raise ValueError('Training manifest changed')
    doc, x, coarse, folds, groups = checked_data(root)
    complete = validate_derivative(read_json(base_root / 'manifest.json'))
    if any(folds[i] == 0 for i in complete):
        raise ValueError('Complete-background review crosses fit boundary')
    targets, skipped = [], []
    for i, row in enumerate(doc['frames']):
        args = (row['truth'], row['negative_regions'])
        control, _ = encode_grid(*args, grid=(15, 20), complete=i in complete)
        if not np.array_equal(control, coarse[i]):
            raise ValueError('Generalized encoding does not reproduce frozen control')
        target, omitted = encode_fine_safe(*args, complete=i in complete)
        if omitted:
            skipped.append(dict(index=i, source=row['source'], regions=omitted))
        targets.append(target)
    return dict(doc, fine_grid_omitted_negatives=skipped), x, np.asarray(targets), folds, groups


def network():
    import keras
    layers = keras.layers
    inputs = layers.Input((120, 160, 3))
    x = layers.Rescaling(1 / 255.)(inputs)
    x = layers.Conv2D(8, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(16, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(24, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(24, 5, padding='same', activation='relu')(x)
    heat = layers.Conv2D(2, 1, bias_initializer=keras.initializers.Constant(-4.), name='center_logits')(x)
    box = layers.Conv2D(4, 1, activation='sigmoid', name='center_geometry')(x)
    return keras.Model(inputs, layers.Concatenate(name='proposal_output')([heat, box]))


def boxes(geometry):
    import tensorflow as tf
    h, w = geometry.shape[1:3]
    xx, yy = tf.meshgrid(tf.range(w, dtype=tf.float32), tf.range(h, dtype=tf.float32))
    center = (geometry[..., :2] + tf.stack([xx, yy], axis=-1)) / tf.constant([float(w), float(h)])
    radius = geometry[..., 2:] / 2
    return tf.clip_by_value(tf.concat([center - radius, center + radius], axis=-1), 0., 1.)


def loss(target, prediction):
    import tensorflow as tf
    from scripts.train_balloon_dense_proposals import sparse_loss
    from scripts.balloon_dense_iou_loss import generalized_iou
    valid = target[..., 8] > 0
    truth = tf.boolean_mask(boxes(target[..., 4:8]), valid)
    predicted = tf.boolean_mask(boxes(prediction[..., 2:]), valid)
    overlap = tf.reduce_sum(1 - generalized_iou(truth, predicted)) / tf.maximum(tf.cast(tf.shape(truth)[0], tf.float32), 1.)
    return sparse_loss(target, prediction) + 2 * overlap


def decode(values):
    if values.shape not in ((15, 20, 6), (30, 40, 6)) or not np.isfinite(values).all():
        raise ValueError('Invalid grid output')
    step = 320 / values.shape[1]
    heat = 1 / (1 + np.exp(-np.clip(values[:, :, :2], -80, 80)))
    pool = []
    for group in (0, 1):
        peaks = ((heat[:, :, group] >= .1)
                 & (heat[:, :, group] == cv2.dilate(heat[:, :, group], np.ones((3, 3), np.uint8))))
        for y, x in np.argwhere(peaks):
            ox, oy, bw, bh = values[y, x, 2:]
            if not np.all((values[y, x, 2:] >= 0) & (values[y, x, 2:] <= 1)):
                raise ValueError('Box channels must be normalized')
            cx, cy = (x + ox) * step, (y + oy) * step
            w, h = bw * 320, bh * 240
            a, b, c, d = max(0., cx - w / 2), max(0., cy - h / 2), min(320., cx + w / 2), min(240., cy + h / 2)
            if min(c - a, d - b) < 4:
                continue
            pad = max(c - a, d - b) * .12
            pool.append(dict(box=[float(a), float(b), float(c), float(d)],
                crop_box=[max(0, int(np.floor(a - pad))), max(0, int(np.floor(b - pad))),
                          min(320, int(np.ceil(c + pad))), min(240, int(np.ceil(d + pad)))],
                area=float((c - a) * (d - b)), color_group=group, proposal_score=float(heat[y, x, group])))
    found = []
    for row in sorted(pool, key=lambda r: -r['proposal_score']):
        if not any(r['color_group'] == row['color_group'] and iou(r['box'], row['box']) > .65 for r in found):
            found.append(row)
        if len(found) == 12:
            break
    return found
