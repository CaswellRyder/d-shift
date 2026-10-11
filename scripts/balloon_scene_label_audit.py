"""Audit TRAIN teacher targets against complete annotations; do not create labels.

Low box IoU does not mean background: a crop may contain a balloon fragment.
Mixed-color crops remain ambiguous even when one localized box matches well.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.balloon_scene_constraints import RAW, RAW_SHA, scene_pixels
from scripts.balloon_wider_crop import wider_crop

OUTPUT = Path('research/balloon-red-blue-20261008/balloon-scene-label-audit-20261011.json')
DETAIL = Path('runs/balloon-scene-label-audit-20261011.json')
LABELS = ('red_balloon', 'blue_balloon')


def checked_box(box):
    values = np.asarray(box, dtype=float)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not 0 <= values[0] < values[2] <= 320
            or not 0 <= values[1] < values[3] <= 240):
        raise ValueError('Require valid 320x240 geometry')
    return values.tolist()


def intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def classify_geometry(box, crop, truth):
    """Conservative annotation category, not an automatically approved crop label."""
    box, crop = checked_box(box), checked_box(crop)
    if not crop[0] <= box[0] < box[2] <= crop[2] or not crop[1] <= box[1] < box[3] <= crop[3]:
        raise ValueError('Localized box must be contained by classifier crop')
    for t in truth:
        checked_box(t['box'])
        if t['label'] not in LABELS:
            raise ValueError('Unsupported complete-scene annotation')
    intersections = [i for i, t in enumerate(truth) if intersection(crop, t['box']) > 0]
    matches = [i for i, t in enumerate(truth) if iou(box, t['box']) >= .5]
    visible = {truth[i]['label'] for i in intersections}
    matched = {truth[i]['label'] for i in matches}
    if not intersections:
        category = 'empty_crop'
    elif len(visible) == 1 and matched == visible:
        category = 'matched_' + next(iter(visible))
    else:
        category = 'partial_or_mixed'
    return dict(category=category, intersecting_truth=intersections, matching_truth=matches,
                maximum_iou=max((iou(box, t['box']) for t in truth), default=0.))


def audit():
    if OUTPUT.exists() or DETAIL.exists():
        raise FileExistsError('Preserve TRAIN annotation audit')
    # Reconstruct every original crop and verify TRAIN sources and annotations.
    pixels, identities = scene_pixels()
    if sha256(RAW) != RAW_SHA or len(pixels) != 385:
        raise ValueError('TRAIN evidence changed')
    frames = read_json(RAW)['training']['parent']['frames']
    records, counts, transitions = [], Counter(), Counter()
    cursor = 0
    for frame in frames:
        for index, found in enumerate(frame['detections']):
            identity = identities[cursor]
            if identity['source'] != frame['source'] or identity['proposal_index'] != index:
                raise ValueError('Proposal identities reordered')
            scores = np.asarray(found['scores'])
            label = int(scores.argmax())
            accepted = bool(label > 0 and scores[label] >= .8)
            if accepted != found['raw_accepted']:
                raise ValueError('Teacher acceptance changed')
            old = classify_geometry(found['box'], found['crop_box'], frame['truth'])
            new_box = wider_crop(found['crop_box'], (240, 320, 3))
            new = classify_geometry(found['box'], new_box, frame['truth'])
            decision = ('background', *LABELS)[label] if accepted else 'not_accepted'
            for mode, geometry in (('original', old), ('wider', new)):
                counts[f'{mode}/{geometry["category"]}/{decision}'] += 1
            transitions[old['category'] + ' -> ' + new['category']] += 1
            records.append(dict(**identity, box=found['box'], truth=frame['truth'],
                original=old, wider=new, wider_crop=new_box,
                teacher_decision=decision, selected=bool(found['accepted'])))
            cursor += 1
    if cursor != 385:
        raise ValueError('Incomplete proposal audit')
    write_json(DETAIL, records)
    write_json(OUTPUT, dict(frames=len(frames), proposals=cursor, counts=dict(sorted(counts.items())),
        geometry_transitions=dict(sorted(transitions.items())),
        raw_training_sha256=RAW_SHA, detail=dict(path=str(DETAIL), sha256=sha256(DETAIL)),
        sources_sha256={str(p): sha256(p) for p in (
            __file__, 'scripts/balloon_scene_constraints.py', 'scripts/balloon_wider_crop.py')},
        scope='Annotation diagnosis only. Geometric categories are not approved training labels. Partial/mixed crops are never treated as background. No threshold tuning or model changes.',
        labels_created=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(counts=dict(sorted(counts.items())), transitions=dict(transitions)), flush=True)


if __name__ == '__main__':
    audit()
