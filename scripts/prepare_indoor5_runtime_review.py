"""Quarantine runtime-shaped crops from already screened TRAIN frames, never IMG_."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from scripts.audit_balloon_dense_proposals import supplement
from scripts.balloon_grid_runtime import GridProposals
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.indoor_expansion5_data import checked_extra, QUEUE, REVIEW, SCREEN
from scripts.prepare_openimages_balloon_review import checked_path, sheets
from scripts.research_balloon_search import experimental

OUTPUT = Path('data/indoor5-runtime-review-20261011')
SEARCH = Path('runs/balloon-grid-convergence-coarse-seed47-20261011/proposal.fp32.tflite')
SEARCH_SHA = '25f71a47c4e8c7682260268b6589284938c7a70bb62eb8392f848c048d28d7c0'


def needs_review(prediction, candidate, positives):
    """Selection only: neither confidence nor absent annotations establish labels."""
    # Reviewed positives carry padded crop bounds, not unpadded object truth.
    # This overlap only queues review; it never supplies a target label or box.
    overlap = max((iou(candidate['crop_box'], r['crop_box']) for r in positives), default=0.)
    return (prediction['accepted'] and prediction['label'] != 'background') or overlap >= .5


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    approved = checked_extra()
    if sha256(MODEL) != MODEL_SHA or sha256(SEARCH) != SEARCH_SHA:
        raise ValueError('Frozen mining model changed')
    source = read_json(QUEUE / 'review.json')
    classifier, grid = Predictor(MODEL, allow_unvalidated=True), GridProposals(SEARCH)
    OUTPUT.mkdir(parents=True)
    for directory in ('frames', 'crops'):
        (OUTPUT / directory).mkdir()
    frames, rows, total = [], [], 0
    for source_id in sorted({r['frame_id'] for r in approved}):
        original = source['frames'][source_id]
        if 'IMG_' in original['source'] or not original['source'].startswith('frame_'):
            raise ValueError('Reserved source family')
        positives = [r for r in approved if r['frame_id'] == source_id and r['label'] != 'background']
        path = checked_path(QUEUE, original['path'], original['sha256'])
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        base = experimental(rgb, 'mser')
        candidates = supplement(base, grid.predict(rgb))
        if candidates[:len(base)] != base or len(candidates) > min(12, len(base) + 2):
            raise ValueError('Runtime proposal budget changed')
        pending = []
        for candidate in candidates:
            total += 1
            a, b, c, d = candidate['crop_box']
            prediction = classifier.predict(rgb[b:d, a:c])
            if needs_review(prediction, candidate, positives):
                pending.append((candidate, prediction))
        if not pending:
            continue
        index = len(frames)
        relative = f'frames/{index:03d}.png'
        Image.fromarray(rgb).save(OUTPUT / relative)
        frames.append(dict(id=index, source_frame_id=source_id, source=original['source'],
            path=relative, sha256=sha256(OUTPUT / relative), source_frame_sha256=original['sha256'],
            boxes=[r['crop_box'] for r in positives], incomplete_positive_annotations=True))
        for candidate, prediction in pending:
            a, b, c, d = candidate['crop_box']
            relative = f'crops/{len(rows):03d}.png'
            Image.fromarray(rgb[b:d, a:c]).save(OUTPUT / relative)
            rows.append(dict(id=len(rows), frame_id=index, source_frame_id=source_id,
                path=relative, sha256=sha256(OUTPUT / relative), source_image='train/' + original['source'],
                box=candidate['box'], crop_box=candidate['crop_box'], prediction=prediction,
                split='train', label='unreviewed'))
    sheets(OUTPUT, 'frames', frames, 3, (320, 270))
    sheets(OUTPUT, 'crops', rows, 6, (160, 160))
    write_json(OUTPUT / 'review.json', dict(frames=frames, samples=rows, proposals_considered=total,
        inputs_sha256={str(p): sha256(p) for p in (QUEUE / 'review.json', REVIEW, SCREEN)},
        models={str(p): dict(sha256=sha256(p), metadata_sha256=sha256(p.with_suffix('.json')))
                for p in (MODEL, SEARCH)},
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/indoor_expansion5_data.py', 'scripts/research_balloon_search.py',
            'scripts/audit_balloon_dense_proposals.py', 'scripts/balloon_grid_runtime.py')},
        sheets={p.name: sha256(p) for p in sorted(OUTPUT.glob('*.jpg'))},
        scope='Review queue only. Same screened TRAIN sources as indoor5, runtime MSER/coarse47 crops. Source annotations incomplete; predictions and absence of overlap are not labels.',
        training_approved=False, test_evaluated=False, deployment_approved=False))
    print(dict(frames=len(frames), crops=len(rows), considered=total,
               queue_sha256=sha256(OUTPUT / 'review.json')))


if __name__ == '__main__':
    prepare()
