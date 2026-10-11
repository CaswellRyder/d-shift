"""Explain failed TRAIN gates by replaying audited scores; never export or read DEV."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from dtr.tracking import iou
from scripts.audit_clutter_round3_feature import OUTPUT as AUDIT
from scripts.balloon_scene_constraints import RAW, RAW_SHA, decisions
from scripts.clutter_round3_feature_data import PROTOCOL
from scripts.blue_object_round3_data import REVIEW, QUEUE
from scripts.research_confident_containment import select
from scripts.train_batch3_retraining import LABELS
from scripts.evaluate_red_blue_development import detection_counts, metrics
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/clutter-round3-feature-diagnosis-20261011.json')


def check_replayed_parent(actual, saved):
    # Keras and the original TFLite scores may differ in floating-point rounding.
    # Decisions, geometry, identities, suppression and counts must still be exact.
    if any(actual[k] != saved[k] for k in ('id', 'source', 'frame_sha256', 'truth', 'counts')):
        raise ValueError('Parent TRAIN identity or counts changed')
    for a, b in zip(actual['detections'], saved['detections'], strict=True):
        keys = ('box', 'crop_box', 'label', 'accepted', 'raw_accepted', 'suppressed',
                'suppressed_by', 'suppressed_by_index', 'rejection_reason', 'suppression_reason')
        if (any(a.get(k) != b.get(k) for k in keys)
                or not np.allclose(a['scores'], b['scores'], atol=1e-5, rtol=0)
                or abs(a['score']-b['score']) > 1e-5):
            raise ValueError('Parent proposal replay differs')


def replay(frames, scores):
    if scores.shape != (sum(len(f['detections']) for f in frames), 3) or not np.isfinite(scores).all():
        raise ValueError('Incomplete or nonfinite proposal scores')
    rows, offset = [], 0
    for frame in frames:
        found = []
        for detection in frame['detections']:
            # Only immutable proposal fields; no previous post-selection state survives.
            row = {k: detection[k] for k in ('box', 'crop_box', 'area', 'color_group', 'proposal_score')
                   if k in detection}
            p = scores[offset]
            label = int(p.argmax())
            row.update(label=LABELS[label], score=float(p[label]), scores=p.tolist(),
                       accepted=bool(label>0 and p[label]>=.8))
            found.append(row)
            offset += 1
        detections = select(found)
        rows.append(dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
            detections=detections, counts=detection_counts(frame['truth'], detections),
            neural_calls=len(found)))
    result = dict(frames=rows, metrics=metrics({c: {k: sum(f['counts'][c][k] for f in rows)
        for k in ('tp', 'fp', 'fn')} for c in LABELS[1:]}))
    checked_metrics(result)
    return result


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = read_json(AUDIT)
    if (audit['script_sha256'] != sha256('scripts/audit_clutter_round3_feature.py')
            or audit['protocol_sha256'] != sha256(PROTOCOL)
            or audit['data_helper_sha256'] != sha256('scripts/clutter_round3_feature_data.py')
            or sha256(RAW) != RAW_SHA
            or any(audit[k] is not False for k in ('development_evaluated', 'test_evaluated',
                                                   'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Audit identity or TRAIN-only scope changed')
    raw = read_json(RAW)
    if set(raw) != {'training'}:
        raise ValueError('Only TRAIN allowed')
    frames = raw['training']['parent']['frames']
    parent = None
    results = {}
    review, queue = read_json(REVIEW), read_json(QUEUE/'review.json')
    if review['queue_sha256'] != sha256(QUEUE/'review.json'):
        raise ValueError('Review changed')
    probes = queue['samples']
    if len(probes) != 13 or any(r['split'] != 'train' for r in probes):
        raise ValueError('Expected13TRAIN backgrounds')
    for name, row in audit['results'].items():
        root = Path(row['run'])
        prov = read_json(root/'provenance.json')
        if (sha256(root/'report.json') != row['report_sha256']
                or sha256(root/'predictions.npz') != row['predictions_sha256']
                or sha256(root/'student.keras') != row['model_sha256']
                or sha256(root/'provenance.json') != row['provenance_sha256']
                or any(sha256(p)!=h for p,h in prov['sources_sha256'].items())):
            raise ValueError('Audited checkpoint changed')
        with np.load(root/'predictions.npz', allow_pickle=False) as stored:
            before, after = stored['before'][-385:], stored['after'][-385:]
            bp, ap = softmax(before), softmax(after)
            original = replay(frames, bp)
            for a, b in zip(original['frames'], frames, strict=True):
                check_replayed_parent(a, b)
            if parent is not None and parent != original:
                raise ValueError('Arms differ in original proposal replay')
            parent = original
            changed = []
            offset = 0
            for frame in frames:
                for index, detection in enumerate(frame['detections']):
                    old, new = int(decisions(before[offset:offset+1])[0]), int(decisions(after[offset:offset+1])[0])
                    identity = prov['scene_identities'][offset]
                    if (identity['source'] != frame['source'] or identity['proposal_index'] != index
                            or identity['crop_box'] != detection['crop_box']):
                        raise ValueError('Proposal order changed')
                    if old != new:
                        changed.append(dict(index=offset, source=frame['source'], box=detection['box'],
                            previous_decision=LABELS[old] if old else 'not_accepted',
                            new_decision=LABELS[new] if new else 'not_accepted',
                            before=bp[offset].tolist(), after=ap[offset].tolist(),
                            maximum_truth_iou={c: max((iou(detection['box'], t['box'])
                                for t in frame['truth'] if t['label']==c), default=0.) for c in LABELS[1:]}))
                    offset += 1
            if [r['index'] for r in changed] != row['scene_changed_rows']:
                raise ValueError('Scene-change gate does not reproduce')
            result = replay(frames, ap)
            result.update(comparison=compare_frames(result['frames'], parent['frames']),
                raw_decision_changes=changed,
                failed_new_backgrounds=[dict(id=probes[i]['id'], source=probes[i]['source_image'],
                    crop_box=probes[i]['proposal']['crop_box'], scores=softmax(stored['probe_after'][i]).tolist())
                    for i, passed in enumerate(row['probe_correct']) if not passed])
            results[name] = result
            print(dict(name=name, metrics=result['metrics'], lost=len(result['comparison']['lost_parent_targets']),
                gained=len(result['comparison']['gained_targets']), changes=changed), flush=True)
    write_json(OUTPUT, dict(parent=parent, results=results, audit_sha256=sha256(AUDIT),
        raw_train_sha256=RAW_SHA, script_sha256=sha256(__file__),
        helpers_sha256={p:sha256(p) for p in ('scripts/research_confident_containment.py',
            'scripts/research_balloon_confidence_selection.py', 'scripts/compare_balloon_containment.py',
            'scripts/audit_indoor5_outcomes.py', 'scripts/summarize_public_balloon_trial.py')},
        interpretation='Failure analysis only: cached audited Keras TRAIN proposal scores reselected at unchanged0.8. Does not override the failed training gate or authorize DEV/export. No new search or hardware execution.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    diagnose()
