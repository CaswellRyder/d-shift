"""TRAIN-only localization/selection diagnosis of newly reviewed blue boxes."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_grid_blue_expansion import checked_run, OUTPUT as TRIAL
from scripts.audit_balloon_dense_proposals import decode, supplement
from scripts.audit_balloon_pipeline_errors import maximum_matching
from scripts.research_balloon_search import experimental

OUTPUT = Path('research/balloon-red-blue-20261008/grid-blue-expansion-diagnosis-20261011.json')


def center_probe(value, box):
    a, b, c, d = box
    x, y = min(19, int((a + c) / 32)), min(14, int((b + d) / 32))
    heat = 1 / (1 + np.exp(-np.clip(value[..., 1], -80, 80)))
    ox, oy, bw, bh = value[y, x, 2:]
    cx, cy = (x + ox) * 16, (y + oy) * 16
    predicted = [max(0., float(cx - bw * 160)), max(0., float(cy - bh * 120)),
                 min(320., float(cx + bw * 160)), min(240., float(cy + bh * 120))]
    overlap = iou(box, predicted)
    peak = bool(heat[y, x] == cv2.dilate(heat, np.ones((3, 3), np.uint8))[y, x])
    return dict(cell=[y, x], heat=float(heat[y, x]), box=predicted, iou=overlap, local_peak=peak)


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial = read_json(TRIAL)
    results = {}
    for seed in (47, 48):
        _, _, probe, _, report, arrays = checked_run('admit', seed)
        if trial['trials'][f'admit{seed}']['report'] != report:
            raise ValueError('Audited checkpoint changed')
        for point, values in arrays.items():
            rows, reasons = [], Counter()
            for index, record in enumerate(probe['records']):
                if index < 129 or not record['truth']:
                    continue
                with Image.open(record['source_path']) as image:
                    base = experimental(np.asarray(image.convert('RGB')), 'mser')
                value = values[index]
                learned = decode(value)
                added = supplement(base, learned)
                truth = record['truth']
                matched = maximum_matching(truth, learned, lambda t, r: r['color_group'] == 1)
                bounded = maximum_matching(truth, added, lambda t, r: r['color_group'] == 1)
                for i, target in enumerate(truth):
                    center = center_probe(value, target['box'])
                    if i in matched:
                        reason = 'localized'
                    elif center['iou'] < .5:
                        reason = 'supervised_center_geometry_below_iou'
                    elif center['heat'] < .1:
                        reason = 'supervised_center_below_heat_threshold'
                    elif not center['local_peak']:
                        reason = 'supervised_center_not_local_peak'
                    else:
                        reason = 'decoder_pruning_or_matching_competition'
                    reasons[reason] += 1
                    rows.append(dict(source=record['source'], target=target, center=center,
                        learned_matched=i in matched, bounded_matched=i in bounded, reason=reason))
            results[f'admit{seed}/{point}'] = dict(counts=dict(reasons), rows=rows,
                bounded_matched=sum(r['bounded_matched'] for r in rows),
                localized_but_not_bounded=sum(r['learned_matched'] and not r['bounded_matched'] for r in rows))
    write_json(OUTPUT, dict(results=results, trial_sha256=sha256(TRIAL), script_sha256=sha256(__file__),
        scope='TRAIN-only stage diagnostics. Center-box errors are a diagnostic attribution, not a causal ablation. One-to-one matching; no development pixels, threshold selection, independent accuracy or Pi claim.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print({name: {k: v for k, v in row.items() if k != 'rows'} for name, row in results.items()})


if __name__ == '__main__':
    diagnose()
