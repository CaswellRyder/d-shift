"""Quarantine real blue-like mistakes from approved FIT/public TRAIN sources."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_balloon_search_stages import training_source
from scripts.balloon_dense_complete import validate_derivative
from scripts.balloon_grid_runtime import GridProposals
from scripts.prepare_grid_clutter_review import BASE, BASE_SHA, guarded_background
from scripts.train_balloon_dense_proposals import checked_data
from scripts.prepare_openimages_balloon_review import checked_path, sheets
from scripts.openimages_blue_expansion import ROOT, REVIEW, checked_pixels
from scripts.research_balloon_search import experimental

CLASSIFIER = Path('runs/balloon-blue-final-fit-admit-seed48-20261011/student.fp32.tflite')
CLASSIFIER_SHA = '9590f22fcf9b82d8bccb75d7f0e3d364e1dd4e2c1431282ef5916e96e7189353'


def source_frames():
    if sha256(BASE / 'manifest.json') != BASE_SHA:
        raise ValueError('Complete FIT dataset changed')
    doc, _, _, folds, _ = checked_data(BASE)
    complete = validate_derivative(doc)
    if len(complete) != 64 or any(folds[i] == 0 for i in complete):
        raise ValueError('Require64 complete FIT frames only')
    for i in sorted(complete):
        row = doc['frames'][i]
        training_source(row['source'])
        path = BASE / row['path']
        with Image.open(path) as im:
            pixels = np.asarray(im.convert('RGB'))
        yield dict(source=row['source'], source_sha256=row['source_sha256'],
            source_path=str(path), source_file_sha256=sha256(path), base_index=i,
            origin='complete_fit', boxes=[t['box'] for t in row['truth']], truth=row['truth']), pixels
    # Recheck frozen public acquisition, license receipt, duplicate references,
    # and selected positive-source identities before considering any other frame.
    checked_pixels()
    review = read_json(REVIEW)
    queue_root = Path(review['queue'])
    queue = read_json(queue_root / 'review.json')
    acquisition = {r['image_id']: r for r in read_json(ROOT / 'acquisition.json')['images']}
    verified = {r['image_id'] for r in read_json(ROOT / 'license-verification.json')['results'] if r['verified'] is True}
    for frame in queue['frames']:
        if frame['image_id'] not in verified:
            continue
        source = acquisition[frame['image_id']]
        if (frame['attribution']['Subset'] != 'train' or frame['reference_hamming'] <= 8
                or (frame['selected_hamming'] is not None and frame['selected_hamming'] <= 8)
                or source['prior_train_hamming'] <= 8 or source['attribution'] != frame['attribution']
                or source['sha256'] != frame['original_sha256']):
            raise ValueError('Unverified public TRAIN identity')
        path = checked_path(queue_root, frame['path'], frame['sha256'])
        original = checked_path(ROOT, frame['original_path'], frame['original_sha256'])
        with Image.open(original) as im:
            rgb = np.asarray(im.convert('RGB'))
        with Image.open(path) as im:
            pixels = np.asarray(im.convert('RGB'))
        if not np.array_equal(pixels, cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)):
            raise ValueError('Public scan does not reproduce source')
        boxes = frame['boxes'] + [r['box'] for r in review['manual_boxes'] if r['frame_id'] == frame['id']]
        yield dict(source='openimages/train/' + frame['image_id'], source_sha256=frame['original_sha256'],
            source_path=str(path), source_file_sha256=frame['sha256'], public_frame_id=frame['id'],
            origin='public_train', boxes=boxes, attribution=frame['attribution'],
            annotation_caveat='Generic balloon annotations may be incomplete; visual review mandatory'), pixels


def select_review(pool, limit=48):
    selected, per_source = [], Counter()
    for row in sorted(pool, key=lambda r: (-r['mining_prediction']['score'], r['source'], r['crop_box'])):
        if len(selected) >= limit:
            break
        if per_source[row['source']] >= 2:
            continue
        selected.append(row)
        per_source[row['source']] += 1
    return selected


def prepare(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha256(CLASSIFIER) != CLASSIFIER_SHA:
        raise ValueError('Mining classifier changed')
    predictor = Predictor(CLASSIFIER, allow_unvalidated=True)
    models, identities = {}, {}
    for seed in (47, 48):
        path = Path(f'runs/balloon-grid-convergence-coarse-seed{seed}-20261011/proposal.fp32.tflite')
        models[f'coarse{seed}'] = GridProposals(path)
        identities[f'coarse{seed}'] = dict(path=str(path), sha256=sha256(path), metadata_sha256=sha256(path.with_suffix('.json')))
    cache, sources, pool, diagnostics = {}, {}, {}, Counter()
    for row, rgb in source_frames():
        source = row['source']
        if source in sources:
            raise ValueError('Repeated source identity')
        sources[source], cache[source] = row, rgb
        diagnostics[row['origin'] + '_frames'] += 1
        candidates = [('mser', c) for c in experimental(rgb, 'mser')]
        candidates += [(name, c) for name, model in models.items() for c in model.predict(rgb)]
        seen = set()
        for name, candidate in candidates:
            diagnostics['proposals'] += 1
            box = candidate['crop_box']
            if not guarded_background(box, [{'box': b} for b in row['boxes']]):
                diagnostics['touches_guarded_balloon'] += 1
                continue
            key = source, tuple(box)
            if key in seen:
                if key in pool:
                    pool[key]['searches'].append(name)
                continue
            seen.add(key)
            a, b, c, d = box
            prediction = predictor.predict(rgb[b:d, a:c])
            diagnostics['safe_crops_classified'] += 1
            if prediction['label'] != 'blue_balloon' or prediction['score'] < .5:
                continue
            pool[key] = dict(source=source, crop_box=box, proposal_box=candidate['box'], searches=[name],
                mining_prediction={k: prediction[k] for k in ('label', 'score', 'scores', 'accepted')})
    selected = select_review(pool.values())
    output.mkdir(parents=True)
    (output / 'frames').mkdir()
    (output / 'crops').mkdir()
    frames, samples, frame_ids = [], [], {}
    for item in selected:
        source = item['source']
        if source not in frame_ids:
            i = len(frames)
            frame_ids[source] = i
            path = f'frames/{i:03d}.png'
            Image.fromarray(cache[source]).save(output / path)
            frames.append(dict(sources[source], id=i, path=path, sha256=sha256(output / path)))
        a, b, c, d = item['crop_box']
        path = f'crops/{len(samples):03d}.png'
        Image.fromarray(cache[source][b:d, a:c]).save(output / path)
        samples.append(dict(item, id=len(samples), frame_id=frame_ids[source], path=path,
            sha256=sha256(output / path), split='train', label='unreviewed'))
    sheets(output, 'frames', frames, 3, (320, 270))
    sheets(output, 'crops', samples, 6, (160, 160))
    write_json(output / 'review.json', dict(frames=frames, samples=samples, models=identities,
        classifier=dict(path=str(CLASSIFIER), sha256=CLASSIFIER_SHA, metadata_sha256=sha256(CLASSIFIER.with_suffix('.json'))),
        base_manifest_sha256=BASE_SHA, public_review_sha256=sha256(REVIEW),
        public_queue_sha256=read_json(REVIEW)['queue_sha256'], diagnostics=dict(diagnostics),
        pool_size=len(pool), script_sha256=sha256(__file__), guard_pixels=16,
        sheets={p.name: sha256(p) for p in sorted(output.glob('*.jpg'))},
        training_approved=False, review_required=True, test_evaluated=False, deployment_approved=False,
        scope='Only complete FIT and license-verified public TRAIN; no IMG-family or evaluation mining. Blue scores select review candidates, never labels. Every crop needs visual review.'))
    print(dict(crops=len(samples), frames=len(frames), pool=len(pool), diagnostics=dict(diagnostics),
        queue_sha256=sha256(output / 'review.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
