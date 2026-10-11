"""Compact attribution and parent-crop baseline for new reviewed TRAIN positives."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.blue_balloon_candidate_data import checked_extra, ROOT, QUEUE, REVIEW
from scripts.compare_current_blue_search import MODEL, MODEL_SHA

OUTPUT = Path('research/balloon-red-blue-20261008/blue-balloon-admission-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = checked_extra()
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Research parent changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    probes = []
    for row in rows:
        with Image.open(row['file']) as image:
            prediction = predictor.predict(np.asarray(image.convert('RGB')))
        probes.append(dict(id=row['id'], frame_id=row['frame_id'], source_image=row['source_image'],
            prediction=prediction, blue_accepted=prediction['label'] == 'blue_balloon' and prediction['score'] >= .8))
    acquisition, queue, licenses = read_json(ROOT / 'acquisition.json'), read_json(QUEUE / 'review.json'), read_json(ROOT / 'license-verification.json')
    write_json(OUTPUT, dict(download_candidates=len(read_json(ROOT / 'selection.json')['selected_image_ids']),
        acquired_photos=len(acquisition['images']), license_verified_photos=sum(r['verified'] is True for r in licenses['results']),
        review_frames=len(queue['frames']), review_crops=len(queue['samples']), admitted_crops=len(rows), admitted_photos=5,
        exact_crops=[{k: r[k] for k in ('id', 'frame_id', 'source_image', 'sha256', 'label', 'box', 'crop_box')} for r in rows],
        attribution={r['source_image']: r['attribution'] for r in rows},
        parent_model=dict(path=str(MODEL), sha256=MODEL_SHA), parent_crop_probes=probes,
        parent_blue_accepted=sum(r['blue_accepted'] for r in probes),
        hashes={str(p): sha256(p) for p in (ROOT / 'acquisition.json', ROOT / 'selection.json',
            ROOT / 'license-verification.json', QUEUE / 'review.json', REVIEW,
            Path('scripts/blue_balloon_candidate_data.py'), Path(__file__), Path('src/dtr/runtime.py'))},
        source_documentation='https://storage.googleapis.com/openimages/web/download_v7.html',
        next_action='Matched TRAIN trial adding these new-photo positives alongside clutter examples; fixed model size, thresholds and held-out boundaries.',
        scope='Exact TRAIN crop admission and parent diagnostic only. No automatic background labels or full-scene metric. Duplicate screening is heuristic; license display is not an ownership warranty.',
        training_approved=True, test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(crops=len(rows), photographs=5, parent_blue_accepted=sum(r['blue_accepted'] for r in probes),
               receipt_sha256=sha256(OUTPUT)))


if __name__ == '__main__':
    audit()
