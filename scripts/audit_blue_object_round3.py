"""Admit reviewed novel TRAIN clutter, not classifier scores as labels."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.acquire_blue_objects_round3 import previous_ids
from scripts.blue_object_round3_data import checked_extra, ROOT, QUEUE, REVIEW
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.fit_blue_prototype_rejection import checked_inputs

OUTPUT = Path('research/balloon-red-blue-20261008/blue-object-round3-admission-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = checked_extra()
    old, _, _, _, _, previous = checked_inputs()
    if ({r['sha256'] for r in rows} & {r['sha256'] for r in old}
            or {r['source_image'] for r in rows} & {r['source_image'] for r in old}
            or sha256(MODEL) != MODEL_SHA):
        raise ValueError('Novelty or classifier changed')
    acquisition = read_json(ROOT/'acquisition.json')
    selection = read_json(ROOT/'selection.json')
    excluded = previous_ids([read_json(p) for p in selection['excluded_acquisitions']])
    if (set(selection['selected']) & excluded or sorted(excluded) != selection['excluded_ids']
            or len(selection['selected']) != 96 or len(acquisition['images']) != 24
            or any(sha256(p) != h for p, h in selection['excluded_acquisitions'].items())):
        raise ValueError('New source selection changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    measured = []
    for r in rows:
        with Image.open(r['file']) as image:
            prediction = predictor.predict(np.asarray(image.convert('RGB')))
        reference = r['mining_prediction']
        if (prediction['label'] != reference['label'] or prediction['accepted'] != reference['accepted']
                or not np.allclose(prediction['scores'], reference['scores'], atol=1e-6, rtol=0)):
            raise ValueError('Mining predictions fail to reproduce')
        measured.append(dict(id=r['id'], source=r['source_image'], sha256=r['sha256'],
            label=r['label'], predicted_label=prediction['label'], score=prediction['score'],
            accepted=prediction['accepted']))
    license_rows = read_json(ROOT/'license-verification.json')['results']
    sources = [REVIEW, ROOT/'acquisition.json', ROOT/'selection.json', ROOT/'license-verification.json',
        QUEUE/'review.json', previous, __file__, 'scripts/blue_object_round3_data.py',
        'scripts/fit_blue_prototype_rejection.py']
    write_json(OUTPUT, dict(rows=measured, crops=len(rows), photos=len({r['source_image'] for r in rows}),
        source_images_screened=len(selection['selected']), source_images_retained=len(acquisition['images']),
        source_licenses_verified=sum(r['verified'] for r in license_rows),
        source_licenses_unverified=sum(not r['verified'] for r in license_rows),
        accepted_false_positives=dict(Counter(r['predicted_label'] for r in measured if r['accepted'])),
        parent_sha256=MODEL_SHA, sources_sha256={str(p): sha256(p) for p in sources},
        training_approved=True, training_executed=False, development_evaluated=False,
        test_evaluated=False, deployment_approved=False, qualification_achieved=False,
        scope='13assistant-reviewed background crops from10novel public TRAIN photos. Per-photo license links, source pixel reconstruction, duplicate and prior-curriculum novelty guards verified. Model errors selected review candidates, never labels. Related crops and error-mined acceptance counts are not independent accuracy. No IMG-family or Pi-image training.'))
    print({k: read_json(OUTPUT)[k] for k in ('crops', 'photos', 'source_licenses_verified',
        'source_licenses_unverified', 'accepted_false_positives')}, flush=True)


if __name__ == '__main__':
    audit()
