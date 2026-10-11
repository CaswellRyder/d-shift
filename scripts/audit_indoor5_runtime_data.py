"""Measure reviewed TRAIN crop difficulty before deciding whether retraining helps."""
from collections import Counter
from pathlib import Path

from PIL import Image
import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.indoor5_runtime_data import checked_extra, QUEUE, REVIEW
from scripts.evaluate_indoor5_trial import OUTPUT as DEVELOPMENT

OUTPUT = Path('research/balloon-red-blue-20261008/indoor5-runtime-admission-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = checked_extra()
    identities = read_json(DEVELOPMENT)['classifiers']
    models = {}
    for key in ('parent', 'admit47', 'admit48'):
        item = identities[key]
        path = Path(item['path'])
        if sha256(path) != item['sha256'] or sha256(path.with_suffix('.json')) != item['metadata_sha256']:
            raise ValueError('Probe model changed')
        models[key] = Predictor(path, allow_unvalidated=True)
    probes, failures = [], {k: [] for k in models}
    for row in rows:
        with Image.open(row['file']) as image:
            rgb = np.asarray(image.convert('RGB'))
        predictions = {k: model.predict(rgb) for k, model in models.items()}
        for key, prediction in predictions.items():
            if prediction['label'] != row['label'] or (row['label'] != 'background' and not prediction['accepted']):
                failures[key].append(dict(id=row['id'], truth=row['label'], prediction=prediction['label'], score=prediction['score']))
        probes.append(dict(id=row['id'], label=row['label'], source=row['source_image'],
            sha256=row['sha256'], predictions=predictions))
    write_json(OUTPUT, dict(rows=probes, class_counts=dict(Counter(r['label'] for r in rows)),
        sources=len({r['source_image'] for r in rows}), failures=failures,
        queue_sha256=sha256(QUEUE / 'review.json'), review_sha256=sha256(REVIEW),
        development_sha256=sha256(DEVELOPMENT), models={k: identities[k] for k in models},
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/indoor5_runtime_data.py', 'scripts/indoor_expansion5_data.py')},
        scope='TRAIN-only crop admission/difficulty probe. Related crops not independent samples. No retraining or full-scene benefit established.',
        training_performed=False, test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print(dict(rows=len(rows), classes=dict(Counter(r['label'] for r in rows)), failures=failures))


if __name__ == '__main__':
    audit()
