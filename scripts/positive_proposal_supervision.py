"""Derive positive crop labels only from reviewed TRAIN geometry, never predictions."""
import numpy as np
from PIL import Image

from dtr.tracking import iou
from scripts.audit_scale_balanced_search import checked_panels
from scripts.public_scene_localizer_data import tensor_hash

LABELS = ('background', 'red_balloon', 'blue_balloon')


def annotation_label(box, truth):
    overlaps = {label: max((iou(box, r['box']) for r in truth if r['label'] == label), default=0.)
                for label in LABELS[1:]}
    qualifying = [label for label in LABELS[1:] if overlaps[label] >= .5 and
                  all(overlaps[other] < .1 for other in LABELS[1:] if other != label)]
    if not qualifying:
        return None
    label = qualifying[0]
    return dict(label=label, best_iou=overlaps[label], other_color_iou=max(
        overlaps[other] for other in LABELS[1:] if other != label))


def proposals(base):
    """Old385proposal rows and novel public MSER crops, bound to current reviewed sources."""
    from dtr.data import read_json, sha256
    from scripts.audit_direct_hybrid_balloon import OUTPUT as HYBRID
    doc = read_json(HYBRID)
    if sha256(doc['raw']['path']) != doc['raw']['sha256']:
        raise ValueError('Public MSER geometry changed')
    saved = read_json(doc['raw']['path'])
    existing = {}
    for i, x in enumerate(base['x']):
        existing.setdefault(tensor_hash(x), []).append(i)
    records, added, duplicate_public = [], [], []
    old_index = 0
    for panel, frames in checked_panels().items():
        for frame_index, (frame, rgb) in enumerate(frames):
            if 'IMG_' in frame['source'] or not frame['source'].startswith(('train/', 'openimages/train/')):
                raise ValueError('Only approved TRAIN frames may contribute')
            source = frame if panel == 'original' else saved[panel]['base_parent']['frames'][frame_index]
            if any(source[k] != frame[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                raise ValueError('Proposal source identity changed')
            for proposal_index, proposal in enumerate(source['detections']):
                index = old_index if panel == 'original' else None
                if panel == 'original':
                    old_index += 1
                label = annotation_label(proposal['box'], frame['truth'])
                if label is None:
                    continue
                a, b, c, d = proposal['crop_box']
                pixels = np.asarray(Image.fromarray(rgb[b:d, a:c]).resize(
                    (64, 64), Image.Resampling.BILINEAR), np.float32)
                digest = tensor_hash(pixels)
                row = dict(panel=panel, source=frame['source'], frame_sha256=frame['frame_sha256'],
                    proposal_index=proposal_index, box=proposal['box'], crop_box=proposal['crop_box'],
                    pixels_sha256=digest, **label)
                if panel == 'original':
                    position = 2222+index
                    identity = base['scene_ids'][index]
                    if identity['source'] != frame['source'] or identity['crop_box'] != proposal['crop_box']:
                        raise ValueError('Old teacher row identity changed')
                    np.testing.assert_array_equal(pixels, base['x'][position])
                    row.update(position=position, scene_row=index)
                    records.append(row)
                elif digest in existing:
                    duplicate_public.append(row)
                else:
                    row.update(position=2639+len(added), scene_row=None)
                    records.append(row)
                    added.append(pixels)
                    existing[digest] = [row['position']]
    if old_index != 385:
        raise ValueError('Original teacher rows incomplete')
    # No reviewed negative or opposite positive can share the same model input.
    assignments = {}
    supervised_positions = {r['position'] for r in records}
    for row in records+duplicate_public:
        digest, label = row['pixels_sha256'], LABELS.index(row['label'])
        if digest in assignments and assignments[digest] != label:
            raise ValueError('Opposite positive labels share pixels')
        assignments[digest] = label
        for i in existing.get(digest, []):
            if i < 2639 and base['y'][i] >= 0 and base['y'][i] != label:
                raise ValueError('Positive input conflicts with existing hard label')
            if i < 2639 and base['scene_mask'][i] and i not in supervised_positions:
                raise ValueError('Identical positive input still has an uncorrected teacher target')
    return records, np.stack(added), duplicate_public
