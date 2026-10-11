"""Fixed cross-product of crop classifiers and proposal searches; development only."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.balloon_grid_runtime import GridProposals
from scripts.audit_balloon_dense_proposals import supplement
from scripts.evaluate_balloon_dense_scenes import CLASSIFIER
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline
from scripts.train_balloon_grid_clutter import checked_protocol, PROTOCOL


def checked_models():
    protocol = checked_protocol()
    models = dict(parent=Predictor(CLASSIFIER, allow_unvalidated=True))
    identities = dict(parent=dict(path=CLASSIFIER, sha256=sha256(CLASSIFIER),
                                 metadata_sha256=sha256(Path(CLASSIFIER).with_suffix('.json'))))
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            run = Path(f'runs/balloon-grid-clutter-{arm}-seed{seed}-20261011')
            path = run / 'student.fp32.tflite'
            provenance = read_json(run / 'provenance.json')
            meta = read_json(path.with_suffix('.json'))
            if (read_json(run / 'status.json')['state'] != 'complete'
                    or provenance['mode'] != arm or provenance['seed'] != seed
                    or provenance['protocol_sha256'] != sha256(PROTOCOL)
                    or provenance['epochs'] != 24 or provenance['ordinary_rows'] != 1487
                    or provenance['hard_rows'] != 44 or provenance['hard_weight'] != 4
                    or provenance['learning_rate'] != .0001 or provenance['batch_size'] != 16
                    or provenance['final_epoch_only'] is not True
                    or meta['sha256'] != sha256(path) or meta['deployment_approved'] is not False
                    or meta['source_student_sha256'] != sha256(run / 'student.keras')
                    or meta['provenance_sha256'] != sha256(run / 'provenance.json')
                    or any(sha256(p) != h for p, h in provenance['sources_sha256'].items())):
                raise ValueError('Training run does not match fixed trial')
            name = f'{arm}{seed}'
            models[name] = Predictor(path, allow_unvalidated=True)
            identities[name] = dict(path=str(path), sha256=sha256(path),
                metadata_sha256=sha256(path.with_suffix('.json')), provenance_sha256=sha256(run / 'provenance.json'),
                report_sha256=sha256(run / 'report.json'))
    if any(m.metadata['threshold'] != .8 for m in models.values()):
        raise ValueError('Classifier operating point changed')
    return models, identities


def evaluate(panel, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if panel not in PANELS:
        raise ValueError('Unknown development panel')
    models, identities = checked_models()
    reference_path = f'runs/balloon-grid-convergence-scenes-{panel}-20261011.json'
    reference = read_json(reference_path)
    review_path = f'configs/{panel}.json'
    frames, review = reviewed_frames('data/' + panel, review_path)
    if (review['threshold'] != .8 or review['matching_iou'] != .5
            or reference['classifier_sha256'] != sha256(CLASSIFIER)
            or reference['review_sha256'] != sha256(review_path)
            or reference['queue_sha256'] != review['queue_sha256']):
        raise ValueError('Development reference or operating point changed')
    searches, search_identities = {}, {}
    for seed in (47, 48):
        name = f'coarse{seed}'
        identity = reference['models'][name]
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Frozen localizer changed')
        searches[name] = GridProposals(path)
        search_identities[name] = identity
    rows = {f'{model}/{search}': [] for model in models for search in ('baseline', *searches)}
    for index, (row, rgb) in enumerate(frames):
        base = experimental(rgb, 'mser')
        proposals = dict(baseline=base)
        proposals.update({name: supplement(base, network.predict(rgb)) for name, network in searches.items()})
        if any(p[:len(base)] != base or len(p) > min(12, len(base) + 2) for p in proposals.values()):
            raise ValueError('Baseline proposals or crop budget changed')
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
        reference_sha256=sha256(reference_path), protocol_sha256=sha256(PROTOCOL),
        queue_sha256=review['queue_sha256'], review_sha256=sha256(review_path),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in helpers},
        classifier_threshold=.8, matching_iou=.5, proposal_threshold=.1,
        selection='split_merged', split_max_aspect=2.42913421076193,
        scope='All predeclared classifiers and searches on repeatedly inspected development panels; not independent qualification or timing benchmark.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print({name: result['metrics'] for name, result in results.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=PANELS, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluate(args.panel, args.output)
