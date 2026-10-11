"""All paired classifiers against three frozen proposal searches; development only."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.balloon_grid_runtime import GridProposals
from scripts.audit_balloon_dense_proposals import supplement
from scripts.export_balloon_mask_convergence import checked_trial, TRIAL, PARENT
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline


def classifiers():
    trial = checked_trial()
    paths = dict(parent=PARENT / 'color.fp32.tflite')
    paths.update({name: Path(row['path']) / 'student.fp32.tflite' for name, row in trial['trials'].items()})
    models, identities = {}, {}
    for name, path in paths.items():
        meta = read_json(path.with_suffix('.json'))
        if name != 'parent':
            evidence = trial['trials'][name]
            if (meta['source_student_sha256'] != evidence['model_sha256']
                    or meta['provenance_sha256'] != evidence['provenance_sha256']
                    or meta['training_trial_sha256'] != sha256(TRIAL)
                    or meta['parity_rows'] != 1531 or not 0 <= meta['max_keras_tflite_delta'] <= .002
                    or meta['export_script_sha256'] != sha256('scripts/export_balloon_mask_convergence.py')
                    or meta['deployment_approved'] is not False):
                raise ValueError('Export no longer matches fixed TRAIN trial')
        models[name] = Predictor(path, allow_unvalidated=True)
        if meta['threshold'] != .8:
            raise ValueError('Classifier operating point changed')
        identities[name] = dict(path=str(path), sha256=sha256(path), metadata_sha256=sha256(path.with_suffix('.json')))
    return models, identities


def evaluate(panel, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if panel not in PANELS:
        raise ValueError('Unknown reviewed panel')
    models, identities = classifiers()
    reference_path = f'runs/balloon-grid-convergence-scenes-{panel}-20261011.json'
    reference = read_json(reference_path)
    review_path = f'configs/{panel}.json'
    frames, review = reviewed_frames('data/' + panel, review_path)
    if (review['threshold'] != .8 or review['matching_iou'] != .5
            or reference['classifier_sha256'] != identities['parent']['sha256']
            or reference['review_sha256'] != sha256(review_path)
            or reference['queue_sha256'] != review['queue_sha256']):
        raise ValueError('Reference panels or operating point changed')
    searches, search_identities = {}, {}
    for seed in (47, 48):
        name = f'coarse{seed}'
        identity = reference['models'][name]
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Localizer changed')
        searches[name] = GridProposals(path)
        search_identities[name] = identity
    rows = {f'{model}/{search}': [] for model in models for search in ('baseline', *searches)}
    for index, (row, rgb) in enumerate(frames):
        base = experimental(rgb, 'mser')
        proposals = dict(baseline=base)
        proposals.update({name: supplement(base, model.predict(rgb)) for name, model in searches.items()})
        if any(p[:len(base)] != base or len(p) > min(12, len(base) + 2) for p in proposals.values()):
            raise ValueError('Search budget changed')
        for model_name, predictor in models.items():
            for search, proposed in proposals.items():
                found = classify_candidates(predictor, rgb, proposed)
                detections = select_detections(found, 'split_merged', 2.42913421076193)
                result = dict(id=row['id'], source=row['source'], frame_sha256=row['sha256'],
                    truth=row['provisional_truth'], detections=detections, extra_calls=len(proposed) - len(base),
                    counts=detection_counts(row['provisional_truth'], detections))
                if model_name == 'parent':
                    check_baseline(result, reference['results'][search]['frames'][index])
                rows[f'{model_name}/{search}'].append(result)
    results = {}
    for name, subset in rows.items():
        counts = {label: {key: sum(r['counts'][label][key] for r in subset) for key in ('tp', 'fp', 'fn')} for label in LABELS}
        results[name] = dict(frames=subset, metrics=metrics(counts))
    helpers = ('scripts/balloon_grid_runtime.py', 'scripts/balloon_fine_grid.py',
        'scripts/audit_balloon_dense_proposals.py', 'scripts/pi_balloon_search_bench.py',
        'scripts/evaluate_red_blue_development.py', 'scripts/research_predictor.py',
        'scripts/research_balloon_search.py', 'src/dtr/runtime.py')
    write_json(output, dict(results=results, classifiers=identities, searches=search_identities,
        reference_sha256=sha256(reference_path), training_trial_sha256=sha256(TRIAL),
        queue_sha256=review['queue_sha256'], review_sha256=sha256(review_path),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in helpers},
        classifier_threshold=.8, matching_iou=.5, proposal_threshold=.1,
        selection='split_merged', split_max_aspect=2.42913421076193,
        scope='All paired final160 models on repeatedly inspected related development panels. Not independent qualification or timing benchmark.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print({name: r['metrics'] for name, r in results.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=PANELS, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluate(args.panel, args.output)
