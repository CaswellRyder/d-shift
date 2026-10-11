"""Fixed selection-only experiment: TRAIN screen then all declared development arms."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.balloon_grid_runtime import GridProposals
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_indoor5_runtime_review import SEARCH, SEARCH_SHA
from scripts.pi_balloon_search_bench import select_detections
from scripts.replay_balloon_suppression import raw
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_public_balloon_trial import checked_metrics

PROTOCOL = Path('configs/balloon-confidence-selection-20261011.json')
RESEARCH = Path('research/balloon-red-blue-20261008')
DEVELOPMENT = RESEARCH/'blue-scene-localizer-development-20261011.json'


def validate_raw_predictions(before, after):
    if len(before) != len(after):
        raise ValueError('Selection added or removed candidates')
    for a, b in zip(before, after, strict=True):
        if (any(a[k] != b[k] for k in ('box', 'crop_box', 'label', 'score', 'scores'))
                or b['raw_accepted'] != a['accepted'] or (b['accepted'] and not a['accepted'])):
            raise ValueError('Selection changed raw predictions or revived rejection')


def variants(frame, found):
    selected = dict(current=select_detections(found, 'split_merged', 2.42913421076193),
                    confidence=confidence_selection(found))
    out = {}
    for key, rows in selected.items():
        validate_raw_predictions(found, rows)
        out[key] = dict(frame, detections=rows, counts=detection_counts(frame['truth'], rows))
    return out


def summarize(rows):
    totals = {c: {k: sum(f['counts'][c][k] for f in rows) for k in ('tp', 'fp', 'fn')} for c in LABELS}
    result = dict(frames=rows, metrics=metrics(totals))
    checked_metrics(result)
    return result


def evaluate(stage):
    out = RESEARCH/f'balloon-confidence-selection-{stage}-20261011.json'
    raw_path = Path(f'runs/balloon-confidence-selection-{stage}-20261011.json')
    if out.exists() or raw_path.exists():
        raise FileExistsError(out)
    if sha256(MODEL) != MODEL_SHA or sha256(SEARCH) != SEARCH_SHA:
        raise ValueError('Frozen model changed')
    design = read_json(PROTOCOL)
    if (design['policies'] != ['current', 'confidence'] or design['localizers'] != ['parent', 'background47']
            or design['classifier_threshold'] != .8 or design['matching_iou'] != .5
            or design['max_proposals'] != 12 or design['extra_blue_proposals'] != 2):
        raise ValueError('Fixed design changed')
    reference = read_json(DEVELOPMENT)
    saved_path = Path(reference['raw']['path'])
    if (sha256(saved_path) != reference['raw']['sha256']
            or reference['script_sha256'] != sha256('scripts/evaluate_blue_scene_localizer.py')
            or reference['classifier_sha256'] != MODEL_SHA or reference['parent_localizer_sha256'] != SEARCH_SHA):
        raise ValueError('Source evaluation changed')
    bg = Path(reference['models']['background47']['path'])
    if (sha256(bg) != reference['models']['background47']['sha256']
            or sha256(bg.with_suffix('.json')) != reference['models']['background47']['metadata_sha256']):
        raise ValueError('Background candidate changed')
    inputs, data, sources = {}, {}, {}
    if stage == 'training':
        predictor = Predictor(MODEL, allow_unvalidated=True)
        if predictor.metadata['threshold'] != .8:
            raise ValueError('Classifier threshold changed')
        localizers = dict(parent=GridProposals(SEARCH), background47=DenseProposals(bg))
        source_rows = [(s, image) for s, image in source_frames() if s['origin'] == 'complete_fit']
        if len(source_rows) != design['training_frames']:
            raise ValueError('TRAIN screen coverage changed')
        data['training'] = {f'{k}/{p}': [] for k in localizers for p in design['policies']}
        for i, (source, rgb) in enumerate(source_rows):
            sources[source['source']] = source
            base = experimental(rgb, 'mser')
            for key, localizer in localizers.items():
                proposed = supplement(base, localizer.predict(rgb))
                found = classify_candidates(predictor, rgb, proposed)
                frame = dict(id=i, source=source['source'], frame_sha256=source['source_file_sha256'],
                    truth=source['truth'], neural_calls=len(found))
                for policy, result in variants(frame, found).items():
                    data['training'][f'{key}/{policy}'].append(result)
    elif stage == 'development':
        train_path = RESEARCH/'balloon-confidence-selection-training-20261011.json'
        train = read_json(train_path)
        if (train['protocol_sha256'] != sha256(PROTOCOL) or train['script_sha256'] != sha256(__file__)
                or any(sha256(p) != h for p, h in train['helpers_sha256'].items())):
            raise ValueError('TRAIN screen changed')
        inputs[str(train_path)] = sha256(train_path)
        for panel, saved in read_json(saved_path).items():
            data[panel] = {f'{k}/{p}': [] for k in design['localizers'] for p in design['policies']}
            for key in design['localizers']:
                if checked_metrics(saved[key]) != reference['panels'][panel]['results'][key]['metrics']:
                    raise ValueError('Source metrics changed')
                for old in saved[key]['frames']:
                    found = raw(old['detections'])
                    frame = {k: old[k] for k in ('id', 'source', 'frame_sha256', 'truth', 'neural_calls')}
                    results = variants(frame, found)
                    check_baseline(results['current'], old)
                    for policy, result in results.items():
                        data[panel][f'{key}/{policy}'].append(result)
    else:
        raise ValueError('Unknown evaluation stage')
    panels, full = {}, {}
    for panel, rows in data.items():
        full[panel] = {key: summarize(frames) for key, frames in rows.items()}
        results = {}
        for key, result in full[panel].items():
            base = full[panel][key.split('/')[0]+'/current']
            result_metrics = checked_metrics(result)
            results[key] = dict(metrics=result_metrics, **compare_frames(result['frames'], base['frames']),
                nonregression=nonregression(checked_metrics(base), result_metrics),
                meets_94=meets_target(result_metrics), neural_calls=sum(r['neural_calls'] for r in result['frames']))
        panels[panel] = dict(frames=len(next(iter(rows.values()))), results=results)
        print(panel, {k: dict(metrics=v['metrics'], lost=len(v['lost_parent_targets'])) for k, v in results.items()}, flush=True)
    write_json(raw_path, full)
    write_json(out, dict(panels=panels, sources=sources, input_receipts=inputs,
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/research_balloon_confidence_selection.py',
            'scripts/prepare_blue_false_alarm_review.py', 'scripts/balloon_dense_runtime.py',
            'scripts/balloon_grid_runtime.py', 'scripts/research_balloon_search.py', 'src/dtr/vision.py',
            'scripts/pi_balloon_search_bench.py', 'scripts/audit_indoor5_outcomes.py',
            'scripts/evaluate_red_blue_development.py', 'scripts/replay_balloon_suppression.py')},
        classifier_sha256=MODEL_SHA, parent_localizer_sha256=SEARCH_SHA, background_localizer_sha256=sha256(bg),
        reference_sha256=sha256(DEVELOPMENT), raw=dict(path=str(raw_path), sha256=sha256(raw_path)),
        scope='TRAIN screen or post-hoc related development replay. Selection-only, same proposals and scores. Not independent qualification or Pi timing.',
        training_performed=False, test_evaluated=False, deployment_approved=False, pi_tested=False, qualification_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('training', 'development'), required=True)
    evaluate(parser.parse_args().stage)
