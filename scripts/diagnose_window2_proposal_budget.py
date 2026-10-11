"""Locate TRAIN target losses before/after the fixed proposal budget; oracle diagnosis only."""
from collections import Counter
from pathlib import Path

import cv2

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.tracking import iou
from scripts.window2_scene_labels import REVIEW, checked_frames
from scripts.research_balloon_search import candidate, select
from scripts.research_blue_bright_search import impossible_geometry, search
from scripts.audit_window2_scene_search import OUTPUT as SCENES, RAW

OUTPUT = Path('research/balloon-red-blue-20261008/window2-proposal-budget-20261011.json')


def inspect_pool(rgb, native):
    red, green, blue = cv2.split(rgb)
    stages = dict(raw_geometry=[], valid_geometry=[], contrast_filtered=[])
    for group, plane in enumerate((cv2.subtract(red, cv2.max(green, blue)),
                                    cv2.subtract(blue, cv2.max(red, green)))):
        detector = cv2.MSER_create(5, 6, 25000)
        if group == 1:
            detector.setPass2Only(True)
        regions, bounds = detector.detectRegions(plane)
        for points, (x, y, width, height) in zip(regions, bounds, strict=True):
            stages['raw_geometry'].append(dict(box=[int(x), int(y), int(x + width), int(y + height)], color_group=group))
            c = candidate(points, group, rgb.shape)
            if c:
                stages['valid_geometry'].append(c)
            if impossible_geometry(int(width), int(height)):
                continue
            reduced = native.reduce(plane, points)
            if len(reduced):
                c = candidate(reduced, group, rgb.shape)
                if c:
                    stages['contrast_filtered'].append(c)
    return stages


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = read_json(SCENES)
    if sha256(RAW) != report['raw']['sha256'] or sha256(REVIEW) != report['inputs_sha256'][str(REVIEW)]:
        raise ValueError('Scene evidence changed')
    prior = {f['id']: f for f in read_json(RAW)['parent']['frames']}
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    frames, causes = [], Counter()
    for frame, rgb in checked_frames():
        pools = inspect_pool(rgb, native)
        chosen = select(pools['contrast_filtered'])
        if chosen != search(rgb, native, True):
            raise ValueError('Instrumented search does not reproduce original proposals')
        pools['selected_mser'] = chosen
        pools['selected_with_localizer'] = prior[frame['id']]['detections']
        targets = []
        for t in frame['truth']:
            best = {k: max((iou(t['box'], p['box']) for p in rows), default=0.) for k, rows in pools.items()}
            cause = next((k for k, value in best.items() if value < .5), 'localized')
            # Later stages may recover geometry via the learned localizer.
            if best['selected_with_localizer'] >= .5:
                cause = 'localized'
            causes[cause] += 1
            rank = None
            for i, p in enumerate(sorted(pools['contrast_filtered'], key=lambda p: -p['proposal_score'])):
                if iou(t['box'], p['box']) >= .5:
                    rank = i + 1
                    break
            targets.append(dict(target=t, best_iou=best, first_loss=cause, first_matching_pool_rank=rank))
        frames.append(dict(id=frame['id'], source=frame['source'], frame_sha256=frame['frame_sha256'],
                           counts={k: len(v) for k, v in pools.items()}, targets=targets))
    write_json(OUTPUT, dict(frames=frames, first_loss_counts=dict(causes),
        inputs_sha256={str(p): sha256(p) for p in (SCENES, RAW, REVIEW, __file__,
            'scripts/window2_scene_labels.py', 'scripts/research_balloon_search.py',
            'scripts/research_blue_bright_search.py', native.path)},
        scope='Ground-truth-guided TRAIN localization diagnosis only. Individual best-IoU coverage is not one-to-one recall or deployable selection. Original pipeline reproduced exactly; no thresholds, budgets or weights changed.',
        training_performed=False, development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(causes), flush=True)


if __name__ == '__main__':
    diagnose()
