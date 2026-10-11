"""Recheck reviewed real-photo admissions and probe frozen crop predictions."""
from collections import Counter
from pathlib import Path

from PIL import Image
import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.blue_balloon_batch3_data import ROOT, QUEUE, REVIEW, checked_extra
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.fit_blue_prototype_rejection import checked_inputs

OUTPUT = Path('research/balloon-red-blue-20261008/blue-balloon-batch3-admission-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    extra = checked_extra()
    prior, _, _, _, _, provenance = checked_inputs()
    if ({r['source_image'] for r in extra} & {r['source_image'] for r in prior}
            or {r['sha256'] for r in extra} & {r['sha256'] for r in prior}):
        raise ValueError('Admission repeats current curriculum sources or crops')
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Frozen classifier changed')
    model = Predictor(MODEL, allow_unvalidated=True)
    predictions = []
    for row in extra:
        with Image.open(row['file']) as im:
            rgb = np.asarray(im.convert('RGB'))
        prediction = model.predict(rgb)
        predictions.append(dict(id=row['id'], frame_id=row['frame_id'], source=row['source_image'],
            source_sha256=row['source_sha256'], crop_sha256=row['sha256'], label=row['label'],
            prediction={k: v for k, v in prediction.items() if k != 'latency_ms'},
            correct_accepted=prediction['accepted'] and prediction['label'] == row['label']))
    acquisition, queue = read_json(ROOT/'acquisition.json'), read_json(QUEUE/'review.json')
    inputs = [REVIEW, ROOT/'acquisition.json', ROOT/'license-verification.json',
              ROOT/'selection.json', QUEUE/'review.json', provenance,
              'scripts/blue_balloon_batch3_data.py', 'scripts/fit_blue_prototype_rejection.py', __file__]
    result = dict(inputs_sha256={str(p): sha256(p) for p in inputs},
        model=dict(path=str(MODEL), sha256=MODEL_SHA),
        attempted_sources=len(read_json(ROOT/'selection.json')['selected_image_ids']),
        downloaded_candidate_sources=len(acquisition['images']), licensed_review_sources=len(queue['frames']),
        rejected_unverified_license=len(queue['rejected']), reviewed_crops=len(queue['samples']),
        admitted_crops=len(extra), admitted_photographs=len({r['source_image'] for r in extra}),
        label_counts=dict(Counter(r['label'] for r in extra)), predictions=predictions,
        parent_correct_accepted=sum(p['correct_accepted'] for p in predictions),
        source_ids_disjoint_from_2074_row_curriculum=True,
        scope='Main-agent visual crop labels; guarded real TRAIN acquisition. Post-label frozen classifier probes are training diagnostics, not an independent accuracy metric. Seven views from four photos, not seven independent scenes. No whole-frame or detector box labels approved.',
        training_approved=True, retrained=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(OUTPUT, result)
    print({k: result[k] for k in ('admitted_crops', 'admitted_photographs', 'label_counts', 'parent_correct_accepted')})
    print(predictions)


if __name__ == '__main__':
    audit()
