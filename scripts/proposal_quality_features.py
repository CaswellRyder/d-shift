"""Frozen localizer feature pooling and bounded learned-proposal quality gate."""
import numpy as np

from dtr.tracking import iou


def features(feature_map, proposal):
    if feature_map.shape != (15, 20, 24) or not np.isfinite(feature_map).all():
        raise ValueError('Require finite frozen feature map')
    a, b, c, d = proposal['box']
    group, score = proposal['color_group'], proposal['proposal_score']
    if not 0 <= a < c <= 320 or not 0 <= b < d <= 240 or group not in (0, 1) or not 0 <= score <= 1:
        raise ValueError('Invalid quality candidate')
    x0, y0 = int(np.floor(a/16)), int(np.floor(b/16))
    x1, y1 = int(np.ceil(c/16)), int(np.ceil(d/16))
    pooled = feature_map[y0:y1, x0:x1].mean(axis=(0, 1))
    return np.concatenate([pooled, np.asarray([(a+c)/640, (b+d)/480, (c-a)/320, (d-b)/240,
        score, float(group == 0), float(group == 1)], np.float32)]).astype(np.float32)


def quality_target(proposal, truth, complete):
    if complete is not True:
        raise ValueError('Incomplete scenes cannot label arbitrary proposal quality')
    label = ('red_balloon', 'blue_balloon')[proposal['color_group']]
    return max((iou(proposal['box'], t['box']) for t in truth if t['label'] == label), default=0.)


def gate(rows, qualities, threshold=.5):
    if threshold != .5 or len(rows) != len(qualities):
        raise ValueError('Require fixed gate and aligned rows')
    found = []
    for row, quality in zip(rows, qualities, strict=True):
        result = dict(row)
        if row.get('proposal_origin') == 'learned':
            if quality is None or not np.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError('Require finite learned-box quality')
            result['box_quality'] = float(quality)
            if row['accepted'] and quality < threshold:
                result.update(accepted=False, suppressed=True, suppression_reason='learned_box_quality')
        elif quality is not None:
            raise ValueError('Do not score or change the crop-classifier path')
        found.append(result)
    return found


def network():
    import keras
    inputs = keras.layers.Input((31,))
    hidden = keras.layers.Dense(8, activation='relu', name='quality_hidden')(inputs)
    output = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(2.),
                                name='quality_iou')(hidden)
    return keras.Model(inputs, output)
