"""Research-only blue veto after selection; never restores suppressed proposals."""
from copy import deepcopy

import numpy as np
from PIL import Image


def mirror_crop(rgb, box):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Require HWC RGB uint8')
    if len(box) != 4 or any(not isinstance(v, (int, np.integer)) for v in box):
        raise ValueError('Require integer crop coordinates')
    a, b, c, d = box
    h, w = rgb.shape[:2]
    if not (0 <= a < c <= w and 0 <= b < d <= h):
        raise ValueError('Crop outside frame')
    pixels = np.asarray(Image.fromarray(rgb[b:d, a:c]).resize((64, 64), Image.Resampling.BILINEAR))
    return pixels[:, ::-1].copy()


def verify_blue(predictor, rgb, detections):
    """One extra call per selected blue, fixed .8 threshold; scores/boxes unchanged."""
    if (predictor.metadata['threshold'] != .8 or predictor.metadata['size'] != 64
            or predictor.metadata['classes'] != ['background', 'red_balloon', 'blue_balloon']):
        raise ValueError('Mirror protocol requires frozen classifier operating point')
    if any('blue_mirror' in d for d in detections):
        raise ValueError('Cannot apply gate twice')
    result = deepcopy(detections)
    for row in result:
        if not row['accepted'] or row['label'] != 'blue_balloon':
            continue
        prediction = predictor.predict(mirror_crop(rgb, row['crop_box']))
        scores = np.asarray(prediction['scores'], dtype=float)
        if scores.shape != (3,) or not np.isfinite(scores).all():
            raise ValueError('Nonfinite or malformed mirror scores')
        keep = prediction['label'] == 'blue_balloon' and prediction['accepted']
        row['blue_mirror'] = dict(label=prediction['label'], score=prediction['score'],
            scores=prediction['scores'], accepted=prediction['accepted'], retained=bool(keep))
        row['accepted'] = bool(keep)
    return result
