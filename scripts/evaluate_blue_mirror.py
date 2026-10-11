"""Replay fixed post-selection mirror consistency, with unchanged baseline checks."""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.research_blue_mirror import verify_blue
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
PROTOCOL = Path('configs/blue-mirror-consistency-20261011.json')
HELPERS = ('scripts/research_blue_mirror.py', 'src/dtr/runtime.py',
    'scripts/evaluate_reviewed_balloon_scenes.py', 'scripts/prepare_blue_false_alarm_review.py',
    'scripts/audit_indoor5_outcomes.py', 'scripts/evaluate_red_blue_development.py',
    'scripts/summarize_public_balloon_trial.py')


def evaluate(stage):
    output = BASE/f'blue-mirror-{stage}-20261011.json'
    raw_path = Path(f'runs/blue-mirror-{stage}-20261011.json')
    if output.exists() or raw_path.exists():
        raise FileExistsError(output)
    reference_path = BASE/f'blue-bright-hybrid-{stage}-20261011.json'
    reference = read_json(reference_path)
    previous = Path(reference['raw']['path'])
    if (sha256(previous) != reference['raw']['sha256'] or sha256(MODEL) != MODEL_SHA
            or reference['classifier_sha256'] != MODEL_SHA
            or reference['classifier_metadata_sha256'] != sha256(MODEL.with_suffix('.json'))
            or any(sha256(p) != h for p, h in reference['helpers_sha256'].items())):
        raise ValueError('Frozen baseline changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    inputs = {str(reference_path): sha256(reference_path), str(previous): sha256(previous)}
    probe_path = BASE/'spatial-dependence-training-20261011.json'
    probe = read_json(probe_path)
    if (probe['script_sha256'] != sha256('scripts/probe_balloon_spatial_dependence.py')
            or probe['classifier']['sha256'] != MODEL_SHA
            or any(sha256(p) != h for p, h in probe['source_receipts'].items())):
        raise ValueError('Training motivation changed')
    inputs[str(probe_path)] = sha256(probe_path)
    if stage == 'training':
        sources = [(s, rgb) for s, rgb in source_frames() if s['origin'] == 'complete_fit']
        pixels = {s['source']: (s['source_file_sha256'], s['truth'], rgb) for s, rgb in sources}
        if len(pixels) != 64:
            raise ValueError('TRAIN coverage changed')
    elif stage == 'development':
        training_path = BASE/'blue-mirror-training-20261011.json'
        training = read_json(training_path)
        if (training['script_sha256'] != sha256(__file__) or training['protocol_sha256'] != sha256(PROTOCOL)
                or any(sha256(p) != h for p, h in training['helpers_sha256'].items())):
            raise ValueError('TRAIN screen changed')
        inputs[str(training_path)] = sha256(training_path)
    else:
        raise ValueError('Unknown stage')
    full, panels, max_delta = {}, {}, 0.
    for panel, saved in read_json(previous).items():
        parent = saved['blue_bright']
        if checked_metrics(parent) != reference['panels'][panel]['blue_bright']['metrics']:
            raise ValueError('Baseline counts changed')
        if stage == 'development':
            sources, review = reviewed_frames('data/'+panel, 'configs/'+panel+'.json')
            if review['threshold'] != .8 or review['matching_iou'] != .5:
                raise ValueError('Operating point changed')
            inputs['configs/'+panel+'.json'] = sha256('configs/'+panel+'.json')
            pixels = {s['source']: (s['sha256'], s['provisional_truth'], rgb) for s, rgb in sources}
        if len(pixels) != len(parent['frames']):
            raise ValueError('Scene coverage mismatch')
        rows = []
        for old in parent['frames']:
            digest, truth, rgb = pixels[old['source']]
            if digest != old['frame_sha256'] or truth != old['truth']:
                raise ValueError('Scene identity changed')
            # Reproduce all original classifier outputs, not only survivors.
            for detection in old['detections']:
                a, b, c, d = detection['crop_box']
                prediction = predictor.predict(rgb[b:d, a:c])
                delta = float(np.max(np.abs(np.asarray(prediction['scores'])-detection['scores'])))
                max_delta = max(max_delta, delta)
                if (delta > 1e-6 or prediction['label'] != detection['label']
                        or prediction['accepted'] != detection['raw_accepted']):
                    raise ValueError('Original classifier output did not reproduce')
            ds = verify_blue(predictor, rgb, old['detections'])
            calls = sum('blue_mirror' in d for d in ds)
            rows.append(dict({k: old[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                detections=ds, counts=detection_counts(truth, ds), extra_mirror_calls=calls,
                neural_calls=old['neural_calls']+calls))
        counts = {c: {k: sum(r['counts'][c][k] for r in rows) for k in ('tp','fp','fn')} for c in LABELS}
        mirrored = dict(frames=rows, metrics=metrics(counts))
        checked_metrics(mirrored)
        full[panel] = dict(parent=parent, mirror=mirrored)
        panels[panel] = dict(parent=parent['metrics'], mirror=mirrored['metrics'],
            **compare_frames(rows, parent['frames']),
            nonregression=nonregression(parent['metrics'], mirrored['metrics']),
            meets_94=meets_target(mirrored['metrics']), frames=len(rows),
            parent_neural_calls=sum(r['neural_calls'] for r in parent['frames']),
            mirror_neural_calls=sum(r['neural_calls'] for r in rows),
            extra_mirror_calls=sum(r['extra_mirror_calls'] for r in rows),
            rejected_blue=sum('blue_mirror' in d and not d['blue_mirror']['retained']
                for r in rows for d in r['detections']))
        print(panel, {k: v for k, v in panels[panel].items() if k != 'colors'}, flush=True)
    write_json(raw_path, full)
    write_json(output, dict(panels=panels, inputs=inputs, raw=dict(path=str(raw_path),sha256=sha256(raw_path)),
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__), classifier_sha256=MODEL_SHA,
        classifier_metadata_sha256=sha256(MODEL.with_suffix('.json')), max_baseline_score_delta=max_delta,
        helpers_sha256={p: sha256(p) for p in HELPERS},
        scope='Frozen TRAIN-motivated blue mirror gate on64 complete FIT scenes then three related repeated development panels. Not independent qualification. Additional calls are workload, not measured Pi latency.',
        test_evaluated=False, training_performed=False, deployment_approved=False,
        qualification_achieved=False, pi_tested=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('training','development'), required=True)
    evaluate(parser.parse_args().stage)
