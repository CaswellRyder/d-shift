"""Attribute saved development false positives geometrically; no threshold selection or mining."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.evaluate_red_blue_development import LABELS, detection_counts
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics


def candidate_key(row):
    return (tuple(row['box']), tuple(row.get('crop_box', [])), row.get('color_group'), row.get('proposal_score'))


def classify_errors(frame, baseline):
    truth, matched, errors = frame['truth'], set(), []
    baseline_keys = {candidate_key(row) for row in baseline['detections']}
    counts = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
    for detection in sorted(frame['detections'], key=lambda row: -row['score']):
        label = detection['label']
        if not detection['accepted'] or label not in LABELS:
            continue
        candidates = [(iou(t['box'], detection['box']), i) for i, t in enumerate(truth)
                      if i not in matched and t['label'] == label]
        overlap, index = max(candidates, default=(0., -1))
        if overlap >= .5:
            matched.add(index)
            counts[label]['tp'] += 1
            continue
        counts[label]['fp'] += 1
        same = max((iou(t['box'], detection['box']) for t in truth if t['label'] == label), default=0.)
        other = max((iou(t['box'], detection['box']) for t in truth if t['label'] != label), default=0.)
        if same >= .5:
            reason = 'duplicate_localized_same_color'
        elif other >= .5:
            reason = 'localized_wrong_color'
        elif max(same, other) > 0:
            reason = 'overlaps_target_below_matching_iou'
        else:
            reason = 'no_annotated_target_overlap'
        errors.append(dict(label=label, box=detection['box'], score=detection['score'],
            origin='baseline_search' if candidate_key(detection) in baseline_keys else 'added_localizer',
            reason=reason, best_same_color_iou=float(same), best_other_color_iou=float(other)))
    for i, target in enumerate(truth):
        if i not in matched:
            counts[target['label']]['fn'] += 1
    if counts != detection_counts(truth, frame['detections']):
        raise ValueError('Attribution differs from canonical matching')
    return errors


def audit(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    panels = {}
    for panel in PANELS:
        path = Path(f'runs/balloon-grid-convergence-scenes-{panel}-20261011.json')
        report = read_json(path)
        baseline = report['results']['baseline']['frames']
        results = {}
        for name, variant in report['results'].items():
            verified_metrics = checked_metrics(variant)
            frames = []
            counts = Counter()
            for frame, old in zip(variant['frames'], baseline, strict=True):
                if any(frame[k] != old[k] for k in ('source', 'frame_sha256', 'truth')):
                    raise ValueError('Baseline frame identity changed')
                errors = classify_errors(frame, old)
                frames.append(dict(source=frame['source'], errors=errors))
                counts.update((r['label'], r['origin'], r['reason']) for r in errors)
            results[name] = dict(metrics=verified_metrics,
                error_counts={'/'.join(k): v for k, v in sorted(counts.items())}, frames=frames)
        panels[panel] = dict(path=str(path), sha256=sha256(path), results=results)
    write_json(output, dict(panels=panels, script_sha256=sha256(__file__),
        scope='Repeated development geometric attribution only. Overlap is association, not proof of object identity. Do not use these frames for TRAIN mining, threshold tuning or qualification.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print({p: {n: r['error_counts'] for n, r in v['results'].items()} for p, v in panels.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
