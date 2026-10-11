"""Retrospective TRAIN output-swap attribution, not a deployable hybrid model."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_edge_crop_localizer import OUTPUT as TRIAL, RAW
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_confident_containment import select
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.research_predictor import classify_candidates
from scripts.summarize_dense_background_trial import check_baseline

OUTPUT = Path('research/balloon-red-blue-20261008/edge-crop-output-attribution-20261011.json')
RAW_OUTPUT = Path('runs/edge-crop-output-attribution-20261011.json')


def diagnose():
    if OUTPUT.exists() or RAW_OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial = read_json(TRIAL)
    if (sha256(RAW) != trial['raw']['sha256'] or sha256(MODEL) != MODEL_SHA
            or trial['script_sha256'] != sha256('scripts/evaluate_edge_crop_localizer.py')):
        raise ValueError('Trial identity changed')
    previous = read_json(RAW)
    networks = {'parent': DenseProposals('runs/blue-head-only-background-20261011/proposal.fp32.tflite')}
    for name, item in trial['exports'].items():
        networks[name] = DenseProposals(item['path'])
        if networks[name].metadata != item['metadata']:
            raise ValueError('Export changed')
    classifier = Predictor(MODEL, allow_unvalidated=True)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    pixels = {s['source']: (s, rgb) for s, rgb in source_frames() if s['origin']=='complete_fit'}
    rows = {f'{name}/{part}': [] for name in networks if name != 'parent'
            for part in ('both', 'heat_only', 'geometry_only')}
    for i, frame in enumerate(previous['parent']['frames']):
        source, rgb = pixels[frame['source']]
        if source['source_file_sha256'] != frame['frame_sha256'] or source['truth'] != frame['truth']:
            raise ValueError('Different TRAIN source')
        base = search(rgb, native, True)
        grids = {}
        for name, model in networks.items():
            model.predict(rgb)
            grids[name] = model.interpreter.get_tensor(model.output['index'])[0]
        for name in networks:
            if name == 'parent':
                continue
            for part in ('both', 'heat_only', 'geometry_only'):
                grid = grids[name].copy()
                if part == 'heat_only':
                    grid[..., 2:] = grids['parent'][..., 2:]
                elif part == 'geometry_only':
                    grid[..., :2] = grids['parent'][..., :2]
                candidates = supplement(base, decode(grid, True))
                detections = select(classify_candidates(classifier, rgb, candidates))
                row = dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                    detections=detections, counts=detection_counts(frame['truth'], detections), neural_calls=len(candidates))
                if part == 'both':
                    check_baseline(row, previous[name]['frames'][i])
                rows[f'{name}/{part}'].append(row)
    full, summary = {}, {}
    for name, frames in rows.items():
        total = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
        full[name] = dict(frames=frames, metrics=metrics(total))
        summary[name] = dict(metrics=full[name]['metrics'], comparison=compare_frames(frames, previous['parent']['frames']))
    write_json(RAW_OUTPUT, full)
    result = dict(trial_sha256=sha256(TRIAL), script_sha256=sha256(__file__), results=summary,
        raw=dict(path=str(RAW_OUTPUT), sha256=sha256(RAW_OUTPUT)),
        scope='Retrospective output-factor swaps on64TRAIN scenes. Hybrid outputs require two networks; not a single deployable model, speed claim or independent accuracy. No tuning or checkpoint selection.',
        development_evaluated=False, test_evaluated=False, deployment_approved=False, qualification_achieved=False)
    write_json(OUTPUT, result)
    print({name: dict(metrics=r['metrics'], lost=len(r['comparison']['lost_parent_targets']),
        gained=len(r['comparison']['gained_targets'])) for name, r in summary.items()}, flush=True)


if __name__ == '__main__':
    diagnose()
