"""TRAIN-only feasibility of existing geometry splitters, capped at two extra crops."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from scripts.window1_runtime_data import checked_batch
from scripts.audit_window1_scene_search import REVIEW
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.research_balloon_search import instance_split_candidates, opening_split_candidates
from scripts.research_predictor import classify_candidates
from scripts.research_confident_containment import select
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.audit_balloon_pipeline_errors import analyze_frame

OUTPUT = Path('research/balloon-red-blue-20261008/window1-split-probe-20261011.json')
RAW = Path('runs/window1-split-probe-20261011.json')


def supplement(base, candidates):
    if len(base) > 12:
        raise ValueError('Unexpected baseline proposal budget')
    rows = list(base)
    for row in sorted(candidates, key=lambda r: -r['proposal_score']):
        if any(r['color_group'] == row['color_group'] and iou(r['box'], row['box']) > .65 for r in rows):
            continue
        rows.append(row)
        if len(rows) - len(base) == 2:
            break
    return rows


def probe():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve TRAIN geometry experiment')
    _, _, frames, sources = checked_batch()
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Frozen classifier changed')
    model = Predictor(MODEL, allow_unvalidated=True)
    if model.metadata['threshold'] != .8:
        raise ValueError('Classifier threshold changed')
    review = read_json(REVIEW)
    root = Path(review['queue'])
    originals = read_json(root/'review.json')
    variants = {'baseline': None, 'watershed': instance_split_candidates, 'opening': opening_split_candidates}
    raw, result = {}, {}
    for name, splitter in variants.items():
        output, diagnostics = [], []
        for f in frames:
            source = originals['frames'][f['id']]
            if source['sha256'] != f['frame_sha256']:
                raise ValueError('TRAIN source mismatch')
            with Image.open(checked_path(root, source['path'], source['sha256'])) as im:
                rgb = np.asarray(im.convert('RGB'))
            # Replay verified, classifier-independent geometry; never use GT to choose proposals.
            base = [{k:d[k] for k in ('box', 'crop_box', 'area', 'color_group', 'proposal_score')}
                    for d in f['detections']]
            pool = splitter(rgb) if splitter else []
            proposed = supplement(base, pool)
            found = select(classify_candidates(model, rgb, proposed))
            row = dict({k:f[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                detections=found, counts=detection_counts(f['truth'], found), neural_calls=len(proposed))
            if name == 'baseline':
                check_replayed_parent(row, f)
            output.append(row)
            diagnostics.append(dict(id=f['id'], split_pool=len(pool), extras=len(proposed)-len(base),
                                    errors=analyze_frame(row)))
        measured = dict(frames=output, metrics=metrics({label:{k:sum(f['counts'][label][k] for f in output)
            for k in ('tp', 'fp', 'fn')} for label in LABELS}))
        checked_metrics(measured)
        raw[name] = measured
        result[name] = dict(metrics=measured['metrics'], frames=diagnostics,
            neural_calls=sum(f['neural_calls'] for f in output),
            geometry_oracle=sum(r['errors']['matched_upper_bounds']['geometry_oracle'] for r in diagnostics))
        print(dict(variant=name, metrics=measured['metrics'], geometry_oracle=result[name]['geometry_oracle'],
                   neural_calls=result[name]['neural_calls']), flush=True)
    sources.update({str(p):sha256(p) for p in (MODEL, __file__, 'scripts/research_balloon_search.py',
        'scripts/research_predictor.py', 'scripts/research_confident_containment.py',
        'scripts/audit_balloon_pipeline_errors.py', 'scripts/summarize_public_balloon_trial.py')})
    if any(sha256(p) != h for p, h in sources.items()):
        raise ValueError('Experiment sources changed')
    write_json(RAW, raw)
    write_json(OUTPUT, dict(results=result, inputs_sha256=sources,
        raw=dict(path=str(RAW), sha256=sha256(RAW)), targets=11, frames=6,
        scope='New public TRAIN photos only. Existing splitters use fixed constants, with at most two extra classifier calls/frame. Original classifier and0.8threshold unchanged. Cached verified baseline geometry, not a fresh end-to-end camera run. Geometry oracle is only a localization ceiling. Old-scene nonregression, held-out generalization and Pi cost remain untested.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    probe()
