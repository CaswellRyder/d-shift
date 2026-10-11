"""Freeze reviewed window1 admission and diagnostic parent predictions."""
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_blue_balloon_batch4 import checked_extra as previous_extra
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.reviewed_balloon_window import load_batch
from scripts.audit_blue_balloon_batch5 import checked_extra as batch5_extra

REVIEW = Path('configs/blue-balloon-window1-review-20261011.json')
REVIEW_SHA = '275f75d5499a45a0fef28c5a90509cddf8915fa3a6119bb3314c4dc20cf66c15'
OUTPUT = Path('research/balloon-red-blue-20261008/blue-balloon-window1-admission-20261011.json')


def checked_extra():
    return load_batch(REVIEW, REVIEW_SHA)


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    extra = checked_extra()
    old, _, _, _, _, provenance = checked_inputs()
    prior = old+previous_extra()+batch5_extra()
    if ({r['source_image'] for r in extra} & {r['source_image'] for r in prior}
            or {r['sha256'] for r in extra} & {r['sha256'] for r in prior}):
        raise ValueError('New batch overlaps earlier curriculum')
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Parent changed')
    model = Predictor(MODEL, allow_unvalidated=True)
    predictions = []
    for row in extra:
        prediction = model.predict(row['file'])
        predictions.append(dict(id=row['id'], frame_id=row['frame_id'], source=row['source_image'],
            source_sha256=row['source_sha256'], crop_sha256=row['sha256'], label=row['label'],
            prediction={k: v for k, v in prediction.items() if k != 'latency_ms'},
            correct_accepted=prediction['accepted'] and prediction['label'] == row['label']))
    review = read_json(REVIEW)
    root, queue = Path(review['root']), Path(review['queue'])
    acquisition, prepared = read_json(root/'acquisition.json'), read_json(queue/'review.json')
    result = dict(inputs_sha256={str(p): sha256(p) for p in (
            REVIEW, root/'acquisition.json', root/'license-verification.json', root/'selection.json',
            queue/'review.json', provenance, __file__, 'scripts/reviewed_balloon_window.py')},
        model=dict(path=str(MODEL), sha256=MODEL_SHA), predictions=predictions,
        attempted_photos=len(read_json(root/'selection.json')['selected_image_ids']),
        screened_photos=len(acquisition['images']), licensed_review_photos=len(prepared['frames']),
        reviewed_crops=len(prepared['samples']), admitted_crops=len(extra),
        admitted_photographs=len({r['source_image'] for r in extra}),
        label_counts=dict(Counter(r['label'] for r in extra)),
        parent_correct_accepted=sum(p['correct_accepted'] for p in predictions),
        training_approved=True, retrained=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False,
        scope='Explicit visual crop labels, source licenses and duplicate screening verified. Disjoint from2074old rows and batches4/5; candidate acquisition excludes all earlier attempted public sources. Parent predictions evaluated after labels were frozen. TRAIN diagnostics only;9correlated views from6photos are not independent validation or full-scene/detector labels.')
    write_json(OUTPUT, result)
    print({k: result[k] for k in ('admitted_crops', 'admitted_photographs', 'label_counts', 'parent_correct_accepted')})
    print(predictions)


if __name__ == '__main__':
    audit()


