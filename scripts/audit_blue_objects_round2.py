"""Source-verified hard blue-background admission; no training or scene accuracy."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.blue_object_round2_data import checked_extra, ROOT, QUEUE, REVIEW
from scripts.compare_current_blue_search import MODEL, MODEL_SHA

OUTPUT = Path('research/balloon-red-blue-20261008/blue-objects-round2-admission-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = checked_extra()
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Parent changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    results = []
    for row in rows:
        with Image.open(row['file']) as image:
            prediction = predictor.predict(np.asarray(image.convert('RGB')))
        old = row['mining_prediction']
        if (prediction['label'] != old['label'] or prediction['accepted'] != old['accepted']
                or np.max(np.abs(np.asarray(prediction['scores']) - old['scores'])) > .001):
            raise ValueError('Mining predictions do not reproduce')
        results.append(dict(id=row['id'], source=row['source_image'], sha256=row['sha256'],
                            label=row['label'], prediction=prediction))
    acquisition = read_json(ROOT / 'acquisition.json')
    licenses = read_json(ROOT / 'license-verification.json')
    selection = read_json(ROOT / 'selection.json')
    if set(selection['selected']) & set(selection['excluded_ids']):
        raise ValueError('Previously downloaded source selected')
    write_json(OUTPUT, dict(rows=results, admitted=len(rows),
        photos=len({r['source_image'] for r in rows}),
        parent_raw_blue_acceptances=sum(r['prediction']['accepted'] for r in results),
        acquisition_summary=dict(selected=len(selection['selected']), retained=len(acquisition['images']),
            screened_out=len(acquisition['screened_out']), licenses_verified=sum(r['verified'] for r in licenses['results']),
            retained_categories=dict(Counter(r['category'] for r in acquisition['images']))),
        inputs_sha256={str(p): sha256(p) for p in (ROOT / 'acquisition.json', ROOT / 'selection.json',
            ROOT / 'attribution.json', ROOT / 'license-verification.json', QUEUE / 'review.json', REVIEW)},
        model=dict(path=str(MODEL), sha256=MODEL_SHA, metadata_sha256=sha256(MODEL.with_suffix('.json'))),
        script_sha256=sha256(__file__), helper_sha256=sha256('scripts/blue_object_round2_data.py'),
        scope='Exact new TRAIN background crops. Mining confidence is measured after human visual labeling. Not independent accuracy or evidence of an improvement.',
        training_performed=False, test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print(dict(admitted=len(rows), photos=len({r['source_image'] for r in rows}),
               false_blue_acceptances=sum(r['prediction']['accepted'] for r in results)))


if __name__ == '__main__':
    audit()
