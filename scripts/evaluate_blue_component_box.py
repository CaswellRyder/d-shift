"""One fixed geometry-only ablation with source-linked TRAIN/development replay."""
import argparse
from collections import Counter
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_pipeline_errors import analyze_frame
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.replay_balloon_suppression import raw
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_balloon_search import experimental
from scripts.research_blue_component_box import refine
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
PROTOCOL = Path('configs/blue-component-box-20261011.json')


def prefix_size(rgb, detections):
    base = experimental(rgb, 'mser')
    if not len(base) <= len(detections) <= min(12, len(base)+2):
        raise ValueError('Proposal budget mismatch')
    for a, b in zip(base, detections):
        if any(a[k] != b[k] for k in ('box', 'crop_box', 'color_group', 'area', 'proposal_score')):
            raise ValueError('Baseline proposal prefix does not reproduce')
    return len(base)


def check_invariants(before, after):
    if len(before) != len(after):
        raise ValueError('Proposal count changed')
    for a, b in zip(before, after, strict=True):
        if any(a[k] != b[k] for k in ('crop_box', 'label', 'score', 'scores', 'accepted')):
            raise ValueError('Geometry correction changed crop or classifier output')


def evaluate(stage):
    output = BASE/f'blue-component-box-{stage}-20261011.json'
    raw_output = Path(f'runs/blue-component-box-{stage}-20261011.json')
    if output.exists() or raw_output.exists():
        raise FileExistsError(output)
    input_path = BASE/f'balloon-confidence-selection-{stage}-20261011.json'
    doc = read_json(input_path)
    source_raw = Path(doc['raw']['path'])
    if (sha256(source_raw) != doc['raw']['sha256']
            or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())
            or doc['script_sha256'] != sha256('scripts/evaluate_balloon_confidence_selection.py')):
        raise ValueError('Prior source/selection evidence changed')
    panels = read_json(source_raw)
    inputs = {str(input_path): sha256(input_path), str(source_raw): sha256(source_raw)}
    if stage == 'training':
        pixel_rows = [(r, rgb) for r, rgb in source_frames() if r['origin'] == 'complete_fit']
        if len(pixel_rows) != 64:
            raise ValueError('TRAIN coverage changed')
        pixels = {r['source']: (r['source_file_sha256'], r['truth'], rgb) for r, rgb in pixel_rows}
    elif stage == 'development':
        train_path = BASE/'blue-component-box-training-20261011.json'
        train = read_json(train_path)
        if (train['protocol_sha256'] != sha256(PROTOCOL) or train['script_sha256'] != sha256(__file__)
                or any(sha256(p) != h for p, h in train['helpers_sha256'].items())):
            raise ValueError('TRAIN screen changed')
        inputs[str(train_path)] = sha256(train_path)
    else:
        raise ValueError('Unknown stage')
    results, full = {}, {}
    for panel, old in panels.items():
        if stage == 'development':
            review_path = 'configs/'+panel+'.json'
            sources, review = reviewed_frames('data/'+panel, review_path)
            if review['threshold'] != .8 or review['matching_iou'] != .5:
                raise ValueError('Evaluation threshold changed')
            inputs[review_path] = sha256(review_path)
            pixels = {r['source']: (r['sha256'], r['provisional_truth'], rgb) for r, rgb in sources}
        old_frames = old['parent/confidence']['frames']
        checked_metrics(old['parent/confidence'])
        if len(pixels) != len(old_frames):
            raise ValueError('Frame coverage mismatch')
        rows = dict(current=[], component=[])
        origins = Counter()
        for prior in old_frames:
            digest, truth, rgb = pixels[prior['source']]
            if digest != prior['frame_sha256'] or truth != prior['truth']:
                raise ValueError('Scene pixels or truth changed')
            found = raw(prior['detections'])
            base_count = prefix_size(rgb, found)
            changed = refine(rgb, found, base_count)
            check_invariants(found, changed)
            for key, predictions in (('current', found), ('component', changed)):
                selected = confidence_selection(predictions)
                frame = {k: prior[k] for k in ('id', 'source', 'frame_sha256', 'truth', 'neural_calls')}
                frame.update(detections=selected, counts=detection_counts(truth, selected), base_count=base_count,
                    changed_boxes=sum(r.get('box_refined', False) for r in predictions))
                if key == 'current':
                    check_baseline(frame, prior)
                    for error in analyze_frame(frame)['false_positives']:
                        origins[(error['label'], 'mser' if error['index'] < base_count else 'learned', error['reason'])] += 1
                rows[key].append(frame)
        full[panel], results[panel] = {}, {}
        for key, frames in rows.items():
            totals = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
            values = metrics(totals)
            full[panel][key] = dict(frames=frames, metrics=values)
            checked_metrics(full[panel][key])
            results[panel][key] = dict(metrics=values, **compare_frames(frames, rows['current']),
                nonregression=nonregression(full[panel]['current']['metrics'], values), meets_94=meets_target(values),
                changed_boxes=sum(f['changed_boxes'] for f in frames), neural_calls=sum(f['neural_calls'] for f in frames))
        results[panel]['false_positive_origins'] = {'/'.join(k): v for k, v in sorted(origins.items())}
        print(panel, {k: dict(metrics=v['metrics'], boxes=v['changed_boxes'], lost=len(v['lost_parent_targets']))
            for k, v in results[panel].items() if k != 'false_positive_origins'}, flush=True)
    write_json(raw_output, full)
    write_json(output, dict(panels=results, inputs=inputs, raw=dict(path=str(raw_output), sha256=sha256(raw_output)),
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/research_blue_component_box.py',
            'scripts/research_balloon_confidence_selection.py', 'scripts/research_balloon_search.py',
            'scripts/prepare_blue_false_alarm_review.py', 'scripts/replay_balloon_suppression.py',
            'scripts/audit_indoor5_outcomes.py', 'scripts/audit_balloon_pipeline_errors.py',
            'scripts/evaluate_red_blue_development.py')},
        scope='Geometry-only replay with source RGB and unchanged crops/classifier scores/call counts. Development-motivated hypothesis, not independent validation. No inference latency measured.',
        test_evaluated=False, pi_tested=False, training_performed=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('training', 'development'), required=True)
    evaluate(parser.parse_args().stage)
