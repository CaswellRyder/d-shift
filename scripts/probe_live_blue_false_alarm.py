"""Desktop replay of visually reviewed Pi crops, not fresh hardware or accuracy evidence."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_pi_live_recheck import OUTPUT as LIVE, MODEL_SHA
from scripts.compare_current_blue_search import MODEL as PARENT, MODEL_SHA as PARENT_SHA
from scripts.export_blue_batch_trial import checked_trial

OUTPUT = Path('research/balloon-red-blue-20261008/pi-blue-false-alarm-probes-20261011.json')
DEPLOYED = Path('runs/balloon-red-blue-photometric-clutter4-admit-seed47-20261010/student.fp32.tflite')


def probe():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial, live = checked_trial(), read_json(LIVE)
    if sha256(DEPLOYED) != MODEL_SHA or sha256(PARENT) != PARENT_SHA:
        raise ValueError('Comparison model changed')
    paths = {'pi_model': DEPLOYED, 'research_parent': PARENT,
             **{k: Path(r['path']) / 'student.fp32.tflite' for k, r in trial['trials'].items()}}
    models = {k: Predictor(p, allow_unvalidated=True) for k, p in paths.items()}
    rows, max_delta = [], 0.
    for policy, evidence in live['runs'].items():
        root = Path(evidence['path'])
        if sha256(root / 'report.json') != evidence['report_sha256']:
            raise ValueError('Live report changed')
        report = read_json(root / 'report.json')
        if sha256(root / 'frames.jsonl') != report['files']['frames.jsonl']:
            raise ValueError('Live rows changed')
        frames = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines()]
        for i in (0, 1, 59):
            path = root / f'{i:04d}-raw.png'
            if sha256(path) != report['files'][path.name]:
                raise ValueError('Sample pixels changed')
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            for d in frames[i]['detections']:
                a, b, c, e = d['crop_box']
                if not (0 <= a < c <= 320 and 0 <= b < e <= 240):
                    raise ValueError('Invalid live crop')
                predictions = {k: model.predict(rgb[b:e, a:c]) for k, model in models.items()}
                old = predictions['pi_model']
                delta = float(np.max(np.abs(np.array(old['scores']) - d['scores'])))
                max_delta = max(max_delta, delta)
                if delta > .001 or old['label'] != d['label'] or old['accepted'] != d['raw_accepted']:
                    raise ValueError('Original live classifier does not reproduce')
                rows.append(dict(policy=policy, frame=i, raw_sha256=sha256(path),
                    crop_box=d['crop_box'], box=d['box'], original_live_scores=d['scores'], predictions=predictions))
    write_json(OUTPUT, dict(rows=rows, crops=len(rows), max_original_pi_score_delta=max_delta,
        predicted_balloon_counts={k: sum(r['predictions'][k]['accepted'] and
            r['predictions'][k]['label'] in ('red_balloon', 'blue_balloon') for r in rows) for k in models},
        models={k: dict(path=str(p), sha256=sha256(p), metadata_sha256=sha256(p.with_suffix('.json'))) for k, p in paths.items()},
        live_receipt_sha256=sha256(LIVE), script_sha256=sha256(__file__),
        scope='One dim-room session, correlated Pi crops from visually inspected frames0/1/59. Desktop replay with original Pi-score reproduction, not a fresh Pi execution, independent scenes or full-frame accuracy. No retraining on these samples.',
        test_evaluated=False, pi_timing_measured=False, deployment_approved=False, qualification_achieved=False))
    print(dict(crops=len(rows), max_delta=max_delta,
        predicted_balloon_counts=read_json(OUTPUT)['predicted_balloon_counts']))


if __name__ == '__main__':
    probe()
