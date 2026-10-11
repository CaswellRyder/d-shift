"""TRAIN-only foreground hue intervention; preserves nonselected pixels exactly."""
import cv2
import numpy as np

from scripts.balloon_rb_swap import select_sources


def recolor(pixels, label):
    pixels = np.asarray(pixels)
    if (pixels.ndim != 3 or pixels.shape[-1] != 3 or label not in (1, 2)
            or pixels.dtype not in (np.dtype('uint8'), np.dtype('float32'))
            or not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255))):
        raise ValueError('Require one RGB0..255 foreground crop')
    hsv = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[..., i].astype(float) for i in range(3)]
    delta = (h+90)%180-90 if label == 1 else h-110
    mask = (np.abs(delta) <= 20) & (s >= 40) & (v >= 32)
    target = 110 if label == 1 else 0
    hsv[..., 0][mask] = np.rint((target+delta[mask]*.25)%180).astype(np.uint8)%180
    altered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(pixels.dtype)
    out = pixels.copy()
    out[mask] = altered[mask]
    return out, 3-label, mask


def select(data, count=18):
    groups = {1:[], 2:[]}
    for i in select_sources(data['rows']):
        label = int(data['y'][i])
        if label == 0:
            continue
        _, _, mask = recolor(data['x'][i], label)
        if mask.mean() >= .08:
            groups[label].append(i)
    if any(len(v) < count for v in groups.values()):
        raise ValueError('Insufficient color-evident TRAIN crops')
    return [v[j] for v in groups.values() for j in np.linspace(0, len(v)-1, count, dtype=int)]
