"""Freeze reviewed new-photo labels and the unchanged research parent's outputs."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.blue_balloon_batch_data import ROOT, QUEUE, REVIEW, checked_extra
from scripts.compare_current_blue_search import MODEL, MODEL_SHA

OUTPUT = Path('research/balloon-red-blue-20261008/blue-balloon-batch2-admission-20261011.json')


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
    acquisition, queue, licenses = map(read_json, (
        ROOT / 'acquisition.json', QUEUE / 'review.json', ROOT / 'license-verification.json'))
    write_json(OUTPUT, dict(
        download_candidates=len(read_json(ROOT / 'selection.json')['selected_image_ids']),
        acquired_photos=len(acquisition['images']),
        license_verified_photos=sum(r['verified'] is True for r in licenses['results']),
        review_frames=len(queue['frames']), review_crops=len(queue['samples']),
        admitted_crops=len(rows), admitted_photos=len({r['source_image'] for r in rows}),
        exact_crops=[{k: r[k] for k in ('id', 'frame_id', 'source_image', 'sha256', 'label', 'box', 'crop_box')} for r in rows],
        attribution={r['source_image']: r['attribution'] for r in rows},
        parent_model=dict(path=str(MODEL), sha256=MODEL_SHA), parent_crop_probes=probes,
        parent_blue_accepted=sum(r['blue_accepted'] for r in probes),
        hashes={str(p): sha256(p) for p in (ROOT / 'acquisition.json', ROOT / 'selection.json',
            ROOT / 'license-verification.json', QUEUE / 'review.json', REVIEW,
            Path('scripts/blue_balloon_batch_data.py'), Path(__file__), Path('src/dtr/runtime.py'))},
        source_documentation='https://storage.googleapis.com/openimages/web/download_v7.html',
        scope='Exact TRAIN crop labels and baseline only. Views within each photograph are correlated. Not scene accuracy, independent qualification or a license warranty.',
        next_action='Use matched training controls, hard labels for teacher-wrong positives, and regression checks before any deployment.',
        training_approved=True, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(admitted=len(rows), parent_accepted=sum(r['blue_accepted'] for r in probes)))


if __name__ == '__main__':
    audit()
