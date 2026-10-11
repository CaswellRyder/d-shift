"""Recount fresh TRAIN scene errors and quarantine localized missed-target crops."""
from pathlib import Path

from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.audit_window1_scene_search import OUTPUT as SOURCE, REVIEW
from scripts.audit_balloon_pipeline_errors import analyze_frame
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.prepare_openimages_balloon_review import checked_path

OUTPUT = Path('research/balloon-red-blue-20261008/blue-balloon-window1-errors-20261011.json')
QUEUE = Path('data/blue-balloon-window1-hard-review-20261011')


def summarize():
    if OUTPUT.exists() or QUEUE.exists():
        raise FileExistsError('Preserve error evidence')
    source = read_json(SOURCE)
    if (sha256(source['raw']['path']) != source['raw']['sha256']
            or any(sha256(p) != h for p, h in source['inputs_sha256'].items())):
        raise ValueError('Scene receipt changed')
    raw = read_json(source['raw']['path'])
    if checked_metrics(raw) != source['metrics']:
        raise ValueError('Scene metrics changed')
    review = read_json(REVIEW)
    root = Path(review['queue'])
    originals = read_json(root/'review.json')
    results, samples = [], []
    QUEUE.mkdir(parents=True)
    for f in raw['frames']:
        errors = analyze_frame(f)
        results.append(errors)
        frame = originals['frames'][f['id']]
        if 'openimages/train/'+frame['image_id'] != f['source'] or frame['sha256'] != f['frame_sha256']:
            raise ValueError('Frame identity changed')
        for error in errors['false_negatives']:
            target = f['truth'][error['index']]
            localized = [d for d in f['detections'] if iou(d['box'], target['box']) >= .5]
            if not localized:
                continue
            best = max(localized, key=lambda d: iou(d['box'], target['box']))
            with Image.open(checked_path(root, frame['path'], frame['sha256'])) as im:
                crop = im.convert('RGB').crop(best['crop_box'])
            path = QUEUE/f'{len(samples):03d}.png'
            crop.save(path)
            samples.append(dict(id=len(samples), path=path.name, sha256=sha256(path),
                frame_id=f['id'], source=f['source'], frame_sha256=f['frame_sha256'], target=target,
                proposal=best, reason=error['reason'], candidate_label=target['label']))
    write_json(QUEUE/'review.json', dict(samples=samples, source_sha256=sha256(SOURCE),
        script_sha256=sha256(__file__), training_approved=False, visual_review_required=True,
        scope='Known TRAIN misses only; geometric matches are candidate labels, not automatic admission.'))
    write_json(OUTPUT, dict(metrics=source['metrics'], frame_errors=results, source_sha256=sha256(SOURCE),
        inputs_sha256={str(p):sha256(p) for p in (__file__, 'scripts/audit_balloon_pipeline_errors.py', REVIEW)},
        hard_queue=dict(path=str(QUEUE), sha256=sha256(QUEUE/'review.json')), hard_crops=len(samples),
        scope='Training-only error attribution. Geometry oracle is an upper bound, not achieved accuracy. No inference threshold change or model fitting.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(errors=results, hard_crops=len(samples)), flush=True)


if __name__ == '__main__':
    summarize()

