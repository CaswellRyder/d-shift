"""Fixed asymmetric search comparison using the current frozen research models."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_grid_runtime import GridProposals
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_indoor5_runtime_review import SEARCH, SEARCH_SHA
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_balloon_search_fast import search_fast
from scripts.research_blue_bright_search import search
from scripts.research_predictor import classify_candidates
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

BASE = Path('research/balloon-red-blue-20261008')
PROTOCOL = Path('configs/blue-bright-hybrid-20261011.json')


def evaluate(stage):
    output = BASE/f'blue-bright-hybrid-{stage}-20261011.json'
    raw_path = Path(f'runs/blue-bright-hybrid-{stage}-20261011.json')
    if output.exists() or raw_path.exists():
        raise FileExistsError(output)
    reference_path = BASE/f'balloon-confidence-selection-{stage}-20261011.json'
    reference = read_json(reference_path)
    previous = Path(reference['raw']['path'])
    if (sha256(previous) != reference['raw']['sha256'] or sha256(MODEL) != MODEL_SHA
            or sha256(SEARCH) != SEARCH_SHA
            or any(sha256(p) != h for p, h in reference['helpers_sha256'].items())):
        raise ValueError('Frozen source changed')
    predictor, localizer = Predictor(MODEL, allow_unvalidated=True), GridProposals(SEARCH)
    native_path = Path('artifacts/balloon-regions-mac-20261008/regions.so')
    native = NativeBalloonRegions(native_path, direct=True)
    if (predictor.metadata['threshold'] != .8
            or read_json(native_path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c')):
        raise ValueError('Classifier or native source changed')
    inputs = {str(reference_path): sha256(reference_path), str(previous): sha256(previous)}
    if stage == 'training':
        sources = [(s, rgb) for s, rgb in source_frames() if s['origin'] == 'complete_fit']
        pixels = {s['source']: (s['source_file_sha256'], s['truth'], rgb) for s, rgb in sources}
        if len(pixels) != 64:
            raise ValueError('TRAIN coverage changed')
    elif stage == 'development':
        training_path = BASE/'blue-bright-hybrid-training-20261011.json'
        training = read_json(training_path)
        if (training['script_sha256'] != sha256(__file__) or training['protocol_sha256'] != sha256(PROTOCOL)
                or any(sha256(p) != h for p, h in training['helpers_sha256'].items())):
            raise ValueError('TRAIN screen changed')
        inputs[str(training_path)] = sha256(training_path)
    else:
        raise ValueError('Unknown stage')
    full, panels = {}, {}
    for panel, saved in read_json(previous).items():
        if stage == 'development':
            sources, review = reviewed_frames('data/'+panel, 'configs/'+panel+'.json')
            if review['threshold'] != .8 or review['matching_iou'] != .5:
                raise ValueError('Operating point changed')
            inputs['configs/'+panel+'.json'] = sha256('configs/'+panel+'.json')
            pixels = {s['source']: (s['sha256'], s['provisional_truth'], rgb) for s, rgb in sources}
        previous_rows = saved['parent/confidence']['frames']
        if len(pixels) != len(previous_rows):
            raise ValueError('Scene coverage mismatch')
        rows = dict(full=[], blue_bright=[])
        for old in previous_rows:
            digest, truth, rgb = pixels[old['source']]
            if digest != old['frame_sha256'] or truth != old['truth']:
                raise ValueError('Scene identity changed')
            reference_base = search_fast(rgb, native)
            learned = localizer.predict(rgb)
            for arm in rows:
                base = search(rgb, native, arm == 'blue_bright')
                if arm == 'full' and base != reference_base:
                    raise ValueError('Geometry prefilter changed proposals')
                proposed = supplement(base, learned)
                ds = confidence_selection(classify_candidates(predictor, rgb, proposed))
                value = dict({k: old[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                    detections=ds, counts=detection_counts(truth, ds), neural_calls=len(proposed),
                    base_calls=len(base), extra_calls=len(proposed)-len(base))
                if arm == 'full':
                    check_baseline(value, old)
                rows[arm].append(value)
        full[panel], panels[panel] = {}, {}
        for arm, values in rows.items():
            counts = {c: {k: sum(r['counts'][c][k] for r in values) for k in ('tp', 'fp', 'fn')} for c in LABELS}
            result = dict(frames=values, metrics=metrics(counts))
            full[panel][arm] = result
            checked_metrics(result)
            panels[panel][arm] = dict(metrics=result['metrics'], **compare_frames(values, rows['full']),
                nonregression=nonregression(full[panel]['full']['metrics'], result['metrics']),
                meets_94=meets_target(result['metrics']), neural_calls=sum(r['neural_calls'] for r in values))
        print(panel, {k: dict(metrics=v['metrics'], calls=v['neural_calls'], lost=len(v['lost_parent_targets']))
            for k, v in panels[panel].items()}, flush=True)
    write_json(raw_path, full)
    write_json(output, dict(panels=panels, inputs=inputs, raw=dict(path=str(raw_path), sha256=sha256(raw_path)),
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__), classifier_sha256=MODEL_SHA,
        localizer_sha256=SEARCH_SHA, classifier_metadata_sha256=sha256(MODEL.with_suffix('.json')),
        localizer_metadata_sha256=sha256(SEARCH.with_suffix('.json')),
        helpers_sha256={p: sha256(p) for p in ('scripts/research_blue_bright_search.py',
            'scripts/research_balloon_search_fast.py', 'scripts/research_balloon_search.py',
            'scripts/research_balloon_confidence_selection.py', 'scripts/prepare_blue_false_alarm_review.py',
            'scripts/balloon_grid_runtime.py', 'scripts/research_predictor.py', 'scripts/audit_balloon_dense_proposals.py',
            'scripts/audit_indoor5_outcomes.py', 'scripts/evaluate_red_blue_development.py')},
        scope='Fixed search-policy ablation on TRAIN and repeatedly used related development scenes. Exact full-search prefilter parity verified separately; blue-only bright pass is not equivalent. Calls are workload, not measured latency.',
        test_evaluated=False, training_performed=False, deployment_approved=False, pi_tested=False,
        qualification_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('training', 'development'), required=True)
    evaluate(parser.parse_args().stage)
