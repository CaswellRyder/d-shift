"""Source-aware tiny quality head; never interpret MSER contrast as probability."""
import numpy as np

from scripts.proposal_quality_features import features

LABELS = ('red_balloon', 'blue_balloon')


def candidate(row):
    origin = row['proposal_origin']
    if origin not in ('mser', 'learned') or row['label'] not in LABELS:
        raise ValueError('Only labeled balloon candidates may enter quality scoring')
    score = row['score']
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Require classification probability, not region contrast')
    return dict(row, color_group=LABELS.index(row['label']), proposal_score=score)


def feature_row(maps, row):
    base = features(maps, candidate(row))
    return np.concatenate([base, np.asarray([row['proposal_origin'] == 'learned', row['proposal_origin'] == 'mser'], np.float32)])


def gate(rows, qualities):
    if len(rows) != len(qualities):
        raise ValueError('Quality rows must align')
    found = []
    for row, quality in zip(rows, qualities, strict=True):
        result = dict(row)
        if row['accepted']:
            candidate(row)
            if quality is None or not np.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError('Require finite accepted-box quality')
            result['box_quality'] = float(quality)
            if quality < .5:
                result.update(accepted=False, suppressed=True, suppression_reason='shared_box_quality')
        elif quality is not None:
            raise ValueError('Do not score rejected candidates')
        found.append(result)
    return found


def network():
    import keras
    inputs = keras.layers.Input((33,))
    hidden = keras.layers.Dense(8, activation='relu', name='quality_hidden')(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(2.),
                                 name='quality_output')(hidden)
    return keras.Model(inputs, outputs)
