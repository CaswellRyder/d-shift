"""Run frozen full-frame search on newly reviewed public TRAIN photos."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.audit_blue_balloon_window1 import checked_extra, OUTPUT as ADMISSION
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_scene_decision_retention import LOCALIZER, LOCALIZER_SHA
from scripts.balloon_dense_runtime import DenseProposals
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_predictor import classify_candidates
from scripts.research_confident_containment import select
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.prepare_openimages_balloon_review import checked_path

REVIEW = Path('configs/blue-balloon-window1-scenes-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/blue-balloon-window1-scenes-20261011.json')
RAW = Path('runs/blue-balloon-window1-scenes-20261011.json')


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve full-scene evidence')
    admitted = checked_extra()
    review = read_json(REVIEW)
    root = Path(review['queue'])
    if sha256(root/'review.json') != review['queue_sha256']:
        raise ValueError('Scene review changed')
    queue = read_json(root/'review.json')
    if ({int(k) for k in review['complete_red_blue_frames']} != {r['frame_id'] for r in admitted}
            or sorted(i for ids in review['complete_red_blue_frames'].values() for i in ids) != sorted(map(int, review['labels']))):
        raise ValueError('Reviewed scene coverage changed')
    if sha256(MODEL) != MODEL_SHA or sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Frozen models changed')
    classifier, localizer = Predictor(MODEL, allow_unvalidated=True), DenseProposals(LOCALIZER)
    if classifier.metadata['threshold'] != .8:
        raise ValueError('Classifier threshold changed')
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    frames = []
    for key, ids in review['complete_red_blue_frames'].items():
        fid = int(key)
        frame = queue['frames'][fid]
        with Image.open(checked_path(root, frame['path'], frame['sha256'])) as im:
            rgb = np.asarray(im.convert('RGB'))
        truth = []
        for i in ids:
            sample = queue['samples'][i]
            if sample['id'] != i or sample['frame_id'] != fid or sample['split'] != 'train':
                raise ValueError('Target identity changed')
            truth.append(dict(label=review['labels'][str(i)], box=sample['box'], annotation_id=i))
        proposed = search(rgb, native, True)
        localizer.predict(rgb)
        learned = decode(localizer.interpreter.get_tensor(localizer.output['index'])[0], True)
        proposed = supplement(proposed, learned)
        found = select(classify_candidates(classifier, rgb, proposed))
        frames.append(dict(id=fid, source='openimages/train/'+frame['image_id'], frame_sha256=frame['sha256'],
                           truth=truth, detections=found, counts=detection_counts(truth, found), neural_calls=len(proposed)))
    result = dict(frames=frames, metrics=metrics({c:{k:sum(f['counts'][c][k] for f in frames)
        for k in ('tp', 'fp', 'fn')} for c in LABELS}))
    checked_metrics(result)
    write_json(RAW, result)
    write_json(OUTPUT, dict(metrics=result['metrics'], frames=len(frames), targets=sum(len(f['truth']) for f in frames),
        raw=dict(path=str(RAW), sha256=sha256(RAW)), admission_sha256=sha256(ADMISSION),
        inputs_sha256={str(p):sha256(p) for p in (REVIEW, root/'review.json', MODEL, LOCALIZER, __file__,
            'scripts/research_blue_bright_search.py', 'scripts/research_blue_peak_blend.py',
            'scripts/research_confident_containment.py', 'scripts/research_predictor.py',
            'scripts/audit_balloon_dense_proposals.py', native.path)},
        scope=review['scope'], training_performed=False, development_evaluated=False,
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(metrics=result['metrics'], per_frame={f['id']:f['counts'] for f in frames}), flush=True)


if __name__ == '__main__':
    audit()

