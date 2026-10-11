"""Quarantine difficult crop predictions from reviewed complete FIT scenes only."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_balloon_search_stages import training_source
from scripts.balloon_dense_complete import validate_derivative
from scripts.balloon_grid_runtime import GridProposals
from scripts.evaluate_balloon_dense_scenes import CLASSIFIER
from scripts.mine_red_blue_negatives import safe_negative
from scripts.prepare_openimages_balloon_review import sheets
from scripts.train_balloon_dense_proposals import checked_data

BASE = Path('data/balloon-dense-complete-training-20261011')
BASE_SHA = '3472020134f07c7e9e0f407f8b2aefe9af20974aae8913d57c14455bc0e84409'


def guarded_background(crop, truth):
    guarded = [[max(0, t['box'][0] - 16), max(0, t['box'][1] - 16),
                min(320, t['box'][2] + 16), min(240, t['box'][3] + 16)] for t in truth]
    return safe_negative(crop, guarded)


def prepare(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha256(BASE / 'manifest.json') != BASE_SHA:
        raise ValueError('Complete TRAIN dataset changed')
    doc, _, _, folds, _ = checked_data(BASE)
    complete = validate_derivative(doc)
    if len(complete) != 64 or any(folds[i] == 0 for i in complete):
        raise ValueError('Require existing complete FIT review, not holdout or development')
    models, identities = {}, {}
    for arm in ('coarse', 'context'):
        for seed in (47, 48):
            name = arm + str(seed)
            path = Path(f'runs/balloon-grid-convergence-{arm}-seed{seed}-20261011/proposal.fp32.tflite')
            models[name] = GridProposals(path)
            identities[name] = dict(path=str(path), sha256=sha256(path), metadata_sha256=sha256(path.with_suffix('.json')))
    predictor = Predictor(CLASSIFIER, allow_unvalidated=True)
    pool, image_cache, diagnostics = {}, {}, Counter()
    for i in sorted(complete):
        row = doc['frames'][i]
        training_source(row['source'])
        with Image.open(BASE / row['path']) as image:
            rgb = np.asarray(image.convert('RGB'))
        image_cache[i] = rgb
        for name, model in models.items():
            for candidate in model.predict(rgb):
                diagnostics['proposals'] += 1
                box = candidate['crop_box']
                if not guarded_background(box, row['truth']):
                    diagnostics['touches_guarded_target'] += 1
                    continue
                key = i, tuple(box)
                if key in pool:
                    pool[key]['localizers'].append(name)
                    diagnostics['duplicate_crop'] += 1
                    continue
                a, b, c, d = box
                prediction = predictor.predict(rgb[b:d, a:c])
                diagnostics['classified_safe_crops'] += 1
                if prediction['label'] not in ('red_balloon', 'blue_balloon') or prediction['score'] < .5:
                    diagnostics['not_object_like'] += 1
                    continue
                pool[key] = dict(base_index=i, source=row['source'], source_sha256=row['source_sha256'],
                    crop_box=box, proposal_box=candidate['box'], localizers=[name],
                    mining_prediction={k: prediction[k] for k in ('label', 'score', 'scores', 'accepted')})
    selected, per_frame = [], Counter()
    for row in sorted(pool.values(), key=lambda r: (-r['mining_prediction']['score'], r['source'], r['crop_box'])):
        if len(selected) >= 48:
            break
        if per_frame[row['base_index']] >= 2:
            continue
        selected.append(row)
        per_frame[row['base_index']] += 1
    output.mkdir(parents=True)
    (output / 'frames').mkdir()
    (output / 'crops').mkdir()
    frames, rows, frame_ids = [], [], {}
    for row in selected:
        index = row['base_index']
        if index not in frame_ids:
            fid = len(frames)
            frame_ids[index] = fid
            path = f'frames/{fid:03d}.png'
            Image.fromarray(image_cache[index]).save(output / path)
            original = doc['frames'][index]
            frames.append(dict(id=fid, base_index=index, path=path, sha256=sha256(output / path),
                source=original['source'], source_sha256=original['source_sha256'], truth=original['truth'],
                boxes=[t['box'] for t in original['truth']]))
        a, b, c, d = row['crop_box']
        path = f'crops/{len(rows):03d}.png'
        Image.fromarray(image_cache[index][b:d, a:c]).save(output / path)
        rows.append(dict(row, id=len(rows), frame_id=frame_ids[index], path=path, sha256=sha256(output / path),
                         split='train', label='unreviewed'))
    sheets(output, 'frames', frames, 3, (320, 265))
    sheets(output, 'crops', rows, 6, (160, 160))
    review = dict(samples=rows, frames=frames, models=identities, complete_indices=sorted(complete),
        base_manifest_sha256=BASE_SHA, complete_review=doc['complete_background'],
        classifier_sha256=sha256(CLASSIFIER), classifier_metadata_sha256=sha256(Path(CLASSIFIER).with_suffix('.json')),
        script_sha256=sha256(__file__), diagnostics=dict(diagnostics), pool_size=len(pool),
        sheets={p.name: sha256(p) for p in sorted(output.glob('*.jpg'))},
        guard_pixels=16, mining_object_score_minimum=.5, max_per_frame=2, maximum_samples=48,
        scope='Only64 approved complete FIT scenes; no IMG-family or grouped-holdout mining. Geometric screening is not a background label: every selected crop requires visual review.',
        training_approved=False, review_required=True, test_evaluated=False, deployment_approved=False)
    write_json(output / 'review.json', review)
    print(dict(crops=len(rows), frames=len(frames), pool=len(pool), diagnostics=dict(diagnostics),
               queue_sha256=sha256(output / 'review.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
