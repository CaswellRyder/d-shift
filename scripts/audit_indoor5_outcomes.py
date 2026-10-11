"""Recount indoor5 scene errors and replay a separately reviewed live smoke check.

No training, hardware access, model promotion, or independent qualification.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_balloon_pipeline_errors import analyze_frame, maximum_matching
from scripts.audit_current_blue_search import matched_truth
from scripts.audit_pi_live_recheck import MODEL_SHA, RUNTIME_SHA
from scripts.evaluate_indoor5_trial import OUTPUT as DEVELOPMENT, RAW
from scripts.export_indoor5_trial import checked_trial
from scripts.pi_balloon_live_schedule_bench import trial_summary
from scripts.probe_live_blue_false_alarm import DEPLOYED
from scripts.review_balloon_live import review
from scripts.review_scheduled_balloon_search import validate_schedule
from scripts.summarize_indoor5_trial import OUTPUT as DECISION
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/indoor5-outcomes-20261011.json')
LIVE = Path('runs/balloon-live-presence-20261011-indoor5')
LIVE_SHA = '4e2f005b6354de6006ec64f12327e3299f2005905f50330565ec61f9ef73c128'
REVIEWED = (0, 1, 2, 3, 5, 6, 7)


def compare_frames(frames, parent):
    """Keep identity-level losses visible even when aggregate counts are equal."""
    rows = [analyze_frame(f) for f in frames]
    lost, gained = [], []
    for row, base in zip(frames, parent, strict=True):
        if any(row[k] != base[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
            raise ValueError('Different comparison frames')
        for destination, ids in ((lost, matched_truth(base) - matched_truth(row)),
                                 (gained, matched_truth(row) - matched_truth(base))):
            destination.extend(dict(source=row['source'], target=row['truth'][i]) for i in sorted(ids))
    colors = {}
    for label in ('red_balloon', 'blue_balloon'):
        total = sum(t['label'] == label for f in frames for t in f['truth'])
        covered = sum(len(maximum_matching([t for t in f['truth'] if t['label'] == label],
                                           f['detections'], lambda t, d: True)) for f in frames)
        colors[label] = dict(targets=total, geometric_oracle_covered=covered,
            geometric_oracle_recall=covered / total if total else None,
            errors={kind: dict(Counter(e['reason'] for r in rows for e in r[kind] if e['label'] == label))
                    for kind in ('false_positives', 'false_negatives')})
    return dict(colors=colors, lost_parent_targets=lost, gained_targets=gained)


def reviewed_indices(report):
    if tuple(report['saved_sample_indices']) != REVIEWED:
        raise ValueError('Saved samples differ from the explicit visual review')
    return REVIEWED


def live_probe(classifiers):
    checked = review(LIVE)
    report = checked['report']
    rows = [json.loads(line) for line in (LIVE / 'frames.jsonl').read_text().splitlines()]
    fresh = {**validate_schedule('periodic', rows), **trial_summary(rows, report['elapsed_s'])}
    if (sha256(LIVE / 'frames.jsonl') != LIVE_SHA or report['model_sha256'] != MODEL_SHA
            or report['runtime_library_sha256'] != RUNTIME_SHA or report['policy'] != 'periodic'
            or report['script_sha256'] != sha256('scripts/pi_balloon_live_schedule_bench.py')
            or report['live_helper_sha256'] != sha256('scripts/pi_balloon_live_bench.py')
            or any(report[k] != value for k, value in fresh.items())
            or report['max_pre_camera_golden_score_delta'] > .001
            or not checked['timing_validation']['temporal_observation_valid']
            or sha256(DEPLOYED) != MODEL_SHA):
        raise ValueError('Changed live smoke-check identity or timing')
    paths = {'deployed': DEPLOYED, **{k: Path(v['path']) for k, v in classifiers.items()}}
    models = {k: Predictor(p, allow_unvalidated=True) for k, p in paths.items()}
    probes, max_delta = [], 0.
    for i in reviewed_indices(report):
        path = LIVE / f'{i:04d}-raw.png'
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        for d in rows[i]['detections']:
            a, b, c, e = d['crop_box']
            if not (0 <= a < c <= 320 and 0 <= b < e <= 240):
                raise ValueError('Invalid crop coordinates')
            predictions = {k: model.predict(rgb[b:e, a:c]) for k, model in models.items()}
            old = predictions['deployed']
            delta = float(np.max(np.abs(np.asarray(old['scores']) - d['scores'])))
            max_delta = max(max_delta, delta)
            if delta > .001 or old['label'] != d['label'] or old['accepted'] != d['raw_accepted']:
                raise ValueError('Original Pi prediction does not reproduce')
            probes.append(dict(frame=i, crop_box=d['crop_box'], raw_sha256=sha256(path),
                               live_accepted=d['accepted'], predictions=predictions))
    return dict(report_sha256=checked['report_sha256'], frames_sha256=LIVE_SHA,
        reviewed_raw_images={f'{i:04d}-raw.png': report['files'][f'{i:04d}-raw.png'] for i in REVIEWED},
        visual_review='Main agent inspected all seven retained raw images: dim room, window/curtain, ceiling and wall; no visible balloons. Unsaved frame4 is not visually labeled.',
        timing_validation=checked['timing_validation'],
        timing={k: report[k] for k in ('frames', 'elapsed_s', 'observed_live_fps', 'sensor_to_result',
            'full_observation_age', 'accepted_counts', 'mode_counts', 'health_before', 'health_after')},
        models={k: dict(path=str(p), sha256=sha256(p), metadata_sha256=sha256(p.with_suffix('.json')))
                for k, p in paths.items()},
        probes=probes, max_original_pi_score_delta=max_delta,
        desktop_raw_balloon_acceptances={k: sum(r['predictions'][k]['accepted'] and
            r['predictions'][k]['label'] in ('red_balloon', 'blue_balloon') for r in probes) for k in models},
        limitations=['Eight-frame presence check, not a sustained or isolated performance benchmark.',
            'Seven correlated retained frames from one empty-room session, not independent target scenes.',
            'Candidate scores are desktop crop replay, not candidate Pi execution or full-frame accuracy.',
            'No training or calibration on these samples.'],
        model_replaced=False, esp32_commands_sent=False, motor_commands_sent=False,
        candidate_pi_tested=False, accuracy_qualified=False)


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checked_trial()
    doc, raw, decision = map(read_json, (DEVELOPMENT, RAW, DECISION))
    if (doc['raw']['sha256'] != sha256(RAW)
            or decision['development_receipt_sha256'] != sha256(DEVELOPMENT)
            or decision['qualification_achieved'] is not False):
        raise ValueError('Changed frozen development or unsupported qualification')
    for item in doc['classifiers'].values():
        path = Path(item['path'])
        if sha256(path) != item['sha256'] or sha256(path.with_suffix('.json')) != item['metadata_sha256']:
            raise ValueError('Classifier changed')
    panels = {}
    for panel, variants in raw.items():
        panels[panel] = {}
        for name, result in variants.items():
            if checked_metrics(result) != doc['panels'][panel]['results'][name]:
                raise ValueError('Metrics do not reproduce')
            parent = variants['parent/' + name.split('/')[1]]['frames']
            panels[panel][name] = dict(metrics=result['metrics'], **compare_frames(result['frames'], parent))
    live = live_probe(doc['classifiers'])
    write_json(OUTPUT, dict(panels=panels, live=live, development_sha256=sha256(DEVELOPMENT),
        decision_sha256=sha256(DECISION), raw_sha256=sha256(RAW), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/audit_balloon_pipeline_errors.py',
            'scripts/audit_current_blue_search.py', 'scripts/summarize_public_balloon_trial.py',
            'scripts/review_balloon_live.py', 'scripts/pi_balloon_live_schedule_bench.py',
            'scripts/review_scheduled_balloon_search.py', 'src/dtr/runtime.py')},
        scope='Repeated related development and separate empty-room camera smoke check. Oracle uses truth and is not achievable accuracy.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print(dict(live_crops=len(live['probes']), live_acceptances=live['desktop_raw_balloon_acceptances']))
    for panel, variants in panels.items():
        print(panel, {k: v['colors'] for k, v in variants.items()
                      if k in ('parent/coarse47', 'admit47/coarse47', 'admit48/coarse47')})


if __name__ == '__main__':
    audit()
