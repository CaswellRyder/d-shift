"""TRAIN-only proposal decision retention; rejected proposals are NOT background labels."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256
from dtr.runtime import softmax
from scripts.balloon_head_constraints import constraints
from scripts.prepare_blue_false_alarm_review import source_frames

RAW = Path('runs/constrained-balloon-head-scenes-20261011.json')
RAW_SHA = '8d485eb538810c24f202acb26090aad70646511c14ed54418bf4a40bffa82a95'


def scene_pixels():
    if sha256(RAW) != RAW_SHA:
        raise ValueError('Frozen TRAIN proposals changed')
    raw = read_json(RAW)
    if set(raw) != {'training'}:
        raise ValueError('No development scene data may enter fitting')
    frames = raw['training']['parent']['frames']
    train = {s['source']: (s, rgb) for s, rgb in source_frames() if s['origin'] == 'complete_fit'}
    if len(train) != 64 or set(train) != {r['source'] for r in frames}:
        raise ValueError('TRAIN source coverage changed')
    pixels, identities = [], []
    for frame in frames:
        source, rgb = train[frame['source']]
        if (source['source_file_sha256'] != frame['frame_sha256'] or source['truth'] != frame['truth']
                or 'IMG_' in frame['source'] or not frame['source'].startswith('train/')):
            raise ValueError('TRAIN boundary or identity changed')
        for index, found in enumerate(frame['detections']):
            a, b, c, d = found['crop_box']
            if not 0 <= a < c <= 320 or not 0 <= b < d <= 240:
                raise ValueError('Invalid crop')
            crop = np.asarray(Image.fromarray(rgb[b:d, a:c]).resize((64, 64), Image.Resampling.BILINEAR), np.float32)
            pixels.append(crop)
            identities.append(dict(source=frame['source'], frame_sha256=frame['frame_sha256'],
                proposal_index=index, crop_box=found['crop_box'], scores=found['scores'],
                raw_accepted=found['raw_accepted']))
    if len(pixels) != 385:
        raise ValueError('TRAIN proposal count changed')
    return np.stack(pixels), identities


def retention_constraints(design, logits, threshold=.8):
    """Preserve accepted class; keep rejected foreground probabilities below threshold.

    For each rejected class, bound its gap to the strongest other parent logit.
    This is a sufficient upper bound, not a background semantic annotation.
    """
    design, logits = np.asarray(design, float), np.asarray(logits, float)
    probs = softmax(logits)
    y = probs.argmax(1)
    accepted = (y > 0) & (probs.max(1) >= threshold)
    a, b, identities = constraints(design, logits, y, accepted, np.zeros(len(y), bool))
    rows, bounds = list(a), list(b)
    for i in np.flatnonzero(~accepted):
        for target in (1, 2):
            rival = max((j for j in range(3) if j != target), key=lambda j: logits[i, j])
            row = np.zeros((design.shape[1], 2))
            row[:, target-1] -= design[i]
            if rival:
                row[:, rival-1] += design[i]
            rows.append(row.ravel())
            bounds.append(logits[i, target]-logits[i, rival]-np.log((threshold-.001)/(1-threshold+.001)))
            identities.append((int(i), -target, rival))
    return np.asarray(rows), np.asarray(bounds), identities


def decisions(logits):
    p = softmax(logits)
    labels = p.argmax(1)
    return np.where((labels > 0) & (p.max(1) >= .8), labels, 0)
