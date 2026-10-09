"""Evaluate refined quality/class outputs on hash-bound development proposals.

Class scores may change in joint mode. Always rebuild ordinary duplicate
suppression from fresh predictions rather than retaining parent suppression.
"""
import argparse
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.vision import suppress_duplicates
from scripts.compare_balloon_containment import confidence_ordered
from scripts.evaluate_balloon_quality import QualityPredictor, panel_frames, quality_ordered
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.research_balloon_search import suppress_confirmed_balloon_parts


def refresh_predictions(predictor, rgb, detections):
    """Reuse proposal geometry only; never reuse stale class or suppression state."""
    rows, deltas, label_changes, acceptance_changes = [], [], 0, 0
    for old in detections:
        a, b, c, d = old['crop_box']
        if (any(type(v) is not int for v in (a, b, c, d))
                or not 0 <= a < c <= 320 or not 0 <= b < d <= 240):
            raise ValueError('Invalid saved crop box')
        scores, quality = predictor.predict(rgb[b:d, a:c])
        previous = np.asarray(old['scores'])
        if previous.shape != (3,) or not np.isfinite(previous).all():
            raise ValueError('Invalid saved class scores')
        best = int(scores.argmax())
        label = predictor.metadata['classes'][best]
        accepted = bool(best != 0 and scores[best] >= .8)
        deltas.append(float(np.abs(scores-previous).max()))
        label_changes += label != old['label']
        acceptance_changes += accepted != old['raw_accepted']
        row = {k: old[k] for k in ('box', 'crop_box', 'area', 'color_group', 'proposal_score') if k in old}
        row.update(label=label, score=float(scores[best]), scores=scores.tolist(),
                   accepted=accepted, box_quality=quality)
        rows.append(row)
    return suppress_duplicates(rows), dict(candidate_count=len(rows),
        max_class_probability_delta=max(deltas, default=0.), label_changes=label_changes,
        raw_acceptance_changes=acceptance_changes)


def evaluate(model_path, source_path, name, panel):
    source = read_json(source_path)
    if source.get('test_evaluated') is not False or source.get('deployment_approved') is not False:
        raise ValueError('Require development-only source report')
    predictor = QualityPredictor(model_path)
    mode = predictor.metadata.get('training_mode')
    if mode not in ('joint', 'head_only'):
        raise ValueError('Require joint-refinement or matched control metadata')
    baseline = source['results'][name]
    if baseline['model_sha256'] != predictor.metadata['source_tflite_sha256'] or baseline['threshold'] != .8:
        raise ValueError('Parent classifier identity or threshold changed')
    pixels = panel_frames(panel, source)
    old = baseline['full_frame']['mser_confirmed_parts']
    if len(old['frames']) != len(pixels) or {f['source'] for f in old['frames']} != set(pixels):
        raise ValueError('Scene coverage mismatch')
    fresh, changes = [], []
    for frame in old['frames']:
        rgb, truth = pixels[frame['source']]
        if frame['truth'] != truth:
            raise ValueError('Reviewed truth changed')
        rows, delta = refresh_predictions(predictor, rgb, frame['detections'])
        fresh.append(dict(source=frame['source'], truth=truth, detections=rows))
        changes.append(dict(source=frame['source'], **delta))
    max_delta = max(c['max_class_probability_delta'] for c in changes)
    if mode == 'head_only' and (max_delta > 1e-5 or any(
            c['label_changes'] or c['raw_acceptance_changes'] for c in changes)):
        raise ValueError('Head-only control changed parent class predictions')
    results = {}
    for kind, select in (('parent_first', suppress_confirmed_balloon_parts),
                         ('confidence_ordered', confidence_ordered), ('quality_ordered', quality_ordered)):
        total = {label: dict(tp=0, fp=0, fn=0) for label in LABELS}
        frames = []
        for frame in fresh:
            detections = select(frame['detections'])
            counts = detection_counts(frame['truth'], detections)
            for label in LABELS:
                for key in ('tp', 'fp', 'fn'):
                    total[label][key] += counts[label][key]
            frames.append(dict(frame, detections=detections, counts=counts))
        results[kind] = dict(metrics=metrics(total), frames=frames)
    if mode == 'head_only' and results['parent_first']['metrics'] != old['metrics']:
        raise ValueError('Control failed to reproduce original scene metrics')
    return dict(results=results, baseline_metrics=old['metrics'], training_mode=mode, panel=panel,
        model_sha256=sha256(model_path), metadata_sha256=sha256(Path(model_path).with_suffix('.json')),
        source_report_sha256=sha256(source_path), source_model=name, class_changes=changes,
        max_class_probability_delta=max_delta, script_sha256=sha256(__file__),
        helper_sha256={p: sha256(p) for p in ('scripts/evaluate_balloon_quality.py',
            'scripts/compare_balloon_containment.py', 'scripts/evaluate_red_blue_development.py',
            'scripts/research_balloon_search.py', 'src/dtr/vision.py')},
        scope='Development-only replay: fixed parent proposals, fresh predictions and ordinary NMS',
        test_evaluated=False, deployment_approved=False, pi_timing_measured=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'source-report', 'source-model', 'output'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--panel', choices=('indoor', 'original'), required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = evaluate(args.model, args.source_report, args.source_model, args.panel)
    write_json(args.output, result)
    print({k: v['metrics'] for k, v in result['results'].items()})
