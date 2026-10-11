"""Freeze reviewed data and measure pre-training errors; no model or deployment changes."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.indoor_expansion5_data import checked_extra, QUEUE, REVIEW, SCREEN, ARCHIVE
from scripts.compare_current_blue_search import MODEL, MODEL_SHA

OUTPUT = Path('research/balloon-red-blue-20261008/indoor-expansion5-admission-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = checked_extra()
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Frozen research parent changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    if predictor.metadata['classes'] != ['background', 'red_balloon', 'blue_balloon']:
        raise ValueError('Taxonomy changed')
    values = []
    for row in rows:
        prediction = predictor.predict(row['file'])
        score = prediction['scores'][predictor.metadata['classes'].index(row['label'])]
        values.append(dict(id=row['id'], source=row['source_image'], label=row['label'],
            crop_sha256=row['sha256'], reviewed_class_probability=float(score),
            hard=bool(score < .8), prediction=prediction))
    write_json(OUTPUT, dict(rows=len(rows), counts=dict(Counter(r['label'] for r in rows)),
        unique_training_images=len({r['source_image'] for r in rows}),
        hard_counts=dict(Counter(r['label'] for r in values if r['hard'])), predictions=values,
        model=dict(path=str(MODEL), sha256=MODEL_SHA, metadata_sha256=sha256(MODEL.with_suffix('.json'))),
        sources_sha256={str(p): sha256(p) for p in (QUEUE / 'review.json', REVIEW, SCREEN, ARCHIVE,
            'scripts/indoor_expansion5_data.py', 'scripts/screen_indoor_expansion5.py', __file__,
            'scripts/build_indoor_balloon_training.py', 'scripts/prepare_openimages_balloon_review.py')},
        reviewed_sheets=read_json(REVIEW)['reviewed_sheets'],
        scope='Newly reviewed crop-level TRAIN data, from existing training recording families. Not source-independent development or final qualification.',
        test_evaluated=False, training_performed=False, deployment_approved=False,
        qualification_achieved=False))
    print(dict(rows=len(rows), hard_counts=dict(Counter(r['label'] for r in values if r['hard']))))


if __name__ == '__main__':
    audit()
