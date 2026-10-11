"""Quarantine new-source blue clutter; previous reviewed sources remain excluded."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.balloon_grid_runtime import GridProposals
from scripts.prepare_blue_false_alarm_review import source_frames, select_review
from scripts.prepare_grid_clutter_review import guarded_background
from scripts.prepare_openimages_balloon_review import sheets
from scripts.research_balloon_search import experimental

OUTPUT = Path('data/blue-clutter-round2-review-20261011')
PREVIOUS = Path('data/blue-false-alarm-review-20261011/review.json')
PREVIOUS_SHA = '1b1f7588d3847d50f6564d8801da62ed1abd5e928e8bc0b9e12908d502470a6a'
MODEL = Path('runs/real-blue-clutter-convergence-admit-seed48-20261011/student.fp32.tflite')
MODEL_SHA = '62ecb93da3d3e6f94ca0e1213520fa04191d0ad5666810e489ba0ed26d4ff3c1'


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(PREVIOUS) != PREVIOUS_SHA or sha256(MODEL) != MODEL_SHA:
        raise ValueError('Frozen mining identities changed')
    previous = read_json(PREVIOUS)
    # Whole-source exclusion also protects every ambiguous prior object, not
    # just its previous crop. No second chance at relabeling ambiguous crops.
    excluded = {r['source'] for r in previous['frames']}
    predictor = Predictor(MODEL, allow_unvalidated=True)
    searches = {}
    for name, identity in previous['models'].items():
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Proposal model changed')
        searches[name] = GridProposals(path)
    sources, pixels, pool, diagnostics = {}, {}, {}, Counter()
    for source, rgb in source_frames():
        key = source['source']
        if key in excluded:
            diagnostics['previous_source_excluded'] += 1
            continue
        if key in sources:
            raise ValueError('Duplicate source')
        sources[key], pixels[key] = source, rgb
        diagnostics[source['origin'] + '_frames'] += 1
        proposals = [('mser', c) for c in experimental(rgb, 'mser')]
        proposals += [(name, c) for name, model in searches.items() for c in model.predict(rgb)]
        seen = set()
        for name, candidate in proposals:
            diagnostics['proposals'] += 1
            box = candidate['crop_box']
            if not guarded_background(box, [{'box': b} for b in source['boxes']]):
                diagnostics['touches_guarded_balloon'] += 1
                continue
            crop_key = key, tuple(box)
            if crop_key in seen:
                if crop_key in pool:
                    pool[crop_key]['searches'].append(name)
                continue
            seen.add(crop_key)
            a, b, c, d = box
            prediction = predictor.predict(rgb[b:d, a:c])
            diagnostics['safe_crops_classified'] += 1
            if prediction['label'] != 'blue_balloon' or prediction['score'] < .5:
                continue
            pool[crop_key] = dict(source=key, crop_box=box, proposal_box=candidate['box'], searches=[name],
                mining_prediction={k: prediction[k] for k in ('label', 'score', 'scores', 'accepted')})
    selected = select_review(pool.values(), limit=48)
    if not selected:
        raise ValueError('No candidates; do not create an empty review')
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'frames').mkdir()
    (OUTPUT / 'crops').mkdir()
    frames, samples, frame_ids = [], [], {}
    for item in selected:
        key = item['source']
        if key not in frame_ids:
            i = len(frames)
            frame_ids[key] = i
            path = f'frames/{i:03d}.png'
            Image.fromarray(pixels[key]).save(OUTPUT / path)
            frames.append(dict(sources[key], id=i, path=path, sha256=sha256(OUTPUT / path)))
        a, b, c, d = item['crop_box']
        path = f'crops/{len(samples):03d}.png'
        Image.fromarray(np.ascontiguousarray(pixels[key][b:d, a:c])).save(OUTPUT / path)
        samples.append(dict(item, id=len(samples), frame_id=frame_ids[key], path=path,
            sha256=sha256(OUTPUT / path), split='train', label='unreviewed'))
    sheets(OUTPUT, 'frames', frames, 3, (320, 270))
    sheets(OUTPUT, 'crops', samples, 6, (160, 160))
    write_json(OUTPUT / 'review.json', dict(frames=frames, samples=samples, models=previous['models'],
        classifier=dict(path=str(MODEL), sha256=MODEL_SHA, metadata_sha256=sha256(MODEL.with_suffix('.json'))),
        previous_queue=dict(path=str(PREVIOUS), sha256=PREVIOUS_SHA), excluded_sources=sorted(excluded),
        diagnostics=dict(diagnostics), pool_size=len(pool), script_sha256=sha256(__file__), guard_pixels=16,
        helper_sha256={p: sha256(p) for p in ('scripts/prepare_blue_false_alarm_review.py',
            'scripts/prepare_grid_clutter_review.py', 'scripts/openimages_blue_expansion.py')},
        sheets={p.name: sha256(p) for p in sorted(OUTPUT.glob('*.jpg'))},
        training_approved=False, review_required=True, test_evaluated=False, deployment_approved=False,
        scope='TRAIN-only new-source review queue. All29 previously reviewed sources excluded, including ambiguous crops. Classifier scores select candidates, not labels. No development image mining.'))
    print(dict(crops=len(samples), frames=len(frames), pool=len(pool), diagnostics=dict(diagnostics),
               queue_sha256=sha256(OUTPUT / 'review.json')))


if __name__ == '__main__':
    prepare()
