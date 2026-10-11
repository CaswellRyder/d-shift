"""TRAIN-only error localization through dense box, heat, peak and selection stages."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_public_scene_localizer import OUTPUT as AUDIT
from scripts.research_blue_peak_blend import cell_box, decode

OUTPUT = Path('research/balloon-red-blue-20261008/public-scene-localizer-cells-20261011.json')


def inspect(values, truth):
    heat = 1/(1+np.exp(-np.clip(values[..., :2], -80, 80)))
    decoded = decode(values, True)
    cells = [(y, x, cell_box(values, y, x)) for y in range(15) for x in range(20)]
    peaks = heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))
    rows = []
    for target in truth:
        group = 0 if target['label'] == 'red_balloon' else 1
        a, b, c, d = target['box']
        gx, gy = min(19, int((a+c)/32)), min(14, int((b+d)/32))
        exact = cell_box(values, gy, gx)
        best = dict(all_cells=0., confident_cells=0., local_peaks=0., decoded=0.)
        for y, x, box in cells:
            if box is None:
                continue
            overlap = iou(target['box'], box)
            best['all_cells'] = max(best['all_cells'], overlap)
            if heat[y, x, group] >= .1:
                best['confident_cells'] = max(best['confident_cells'], overlap)
                if peaks[y, x, group]:
                    best['local_peaks'] = max(best['local_peaks'], overlap)
        best['decoded'] = max((iou(target['box'], row['box']) for row in decoded if row['color_group'] == group), default=0.)
        reason = next((k for k, value in best.items() if value < .5), 'localized')
        if best['decoded'] >= .5:
            reason = 'localized'
        rows.append(dict(target=target, center_cell=[gy, gx], center_probability=float(heat[gy, gx, group]),
            center_box=exact, center_box_iou=iou(target['box'], exact) if exact is not None else 0.,
            best_iou=best, first_loss=reason))
    return rows


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = read_json(AUDIT)
    if any(sha256(path) != h for path, h in report['inputs_sha256'].items()):
        raise ValueError('Audited sources changed')
    results = {}
    for name, row in report['training'].items():
        root = Path(row['run'])
        for filename, h in row['artifacts_sha256'].items():
            if sha256(root/filename) != h:
                raise ValueError('Audited predictions changed')
        prov = read_json(root/'provenance.json')
        pred = np.load(root/'predictions.npy', allow_pickle=False)
        if pred.shape != (139, 15, 20, 6) or len(prov['additional_frames']) != 18:
            raise ValueError('Public TRAIN prediction indexing changed')
        frames = []
        for frame, values in zip(prov['additional_frames'], pred[121:], strict=True):
            frames.append(dict(frame, targets=inspect(values, frame['truth'])))
        all_targets = [t for f in frames for t in f['targets']]
        results[name] = dict(frames=frames, targets=len(all_targets),
            first_loss_counts=dict(Counter(t['first_loss'] for t in all_targets)),
            correct_center_boxes=int(sum(t['center_box_iou'] >= .5 for t in all_targets)),
            confident_center_cells=int(sum(t['center_probability'] >= .1 for t in all_targets)))
        print(dict(name=name, **{k: v for k, v in results[name].items() if k != 'frames'}), flush=True)
    write_json(OUTPUT, dict(results=results, audit_sha256=sha256(AUDIT),
        script_sha256=sha256(__file__), decoder_sha256=sha256('scripts/research_blue_peak_blend.py'),
        scope='Ground-truth-guided diagnosis of saved18publicTRAIN predictions,40targets. Each-stage best-IoU is an oracle, not one-to-one detection recall. No fresh fitting, model selection on heldout data or qualification claim.',
        original_holdout_evaluated=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    diagnose()
