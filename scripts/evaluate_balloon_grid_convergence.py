"""Fixed full-scene development comparison of every TRAIN-eligible continuation arm."""
import argparse
from pathlib import Path
import time

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


def evaluate(panel, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if panel not in PANELS:
        raise ValueError('Unknown reviewed development panel')
    trial_path = 'runs/balloon-grid-convergence-training-20261011.json'
    trial = read_json(trial_path)
    reference_path = f'runs/balloon-dense-public-admit-scenes-{panel}-20261011.json'
    reference = read_json(reference_path)
    review_path = f'configs/{panel}.json'
    frames, review = reviewed_frames('data/' + panel, review_path)
    if (review['threshold'] != .8 or review['matching_iou'] != .5
            or sha256(CLASSIFIER) != reference['classifier_sha256']
            or review['queue_sha256'] != reference['queue_sha256']
            or sha256(review_path) != reference['review_sha256']):
        raise ValueError('Fixed operating point or evaluation sources changed')
    predictor = Predictor(CLASSIFIER, allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Classifier threshold changed')
    networks, identities = {}, {}
    for arm in trial['eligible_arms']:
        if not all(r['training_gate_pass'] for r in trial['trials'][arm].values()):
            raise ValueError('Ineligible arm listed')
        for seed in (47, 48):
            path = Path(f'runs/balloon-grid-convergence-{arm}-seed{seed}-20261011/proposal.fp32.tflite')
            model = GridProposals(path)
            m = model.metadata
            expected = trial['trials'][arm][str(seed)]
            if (m['trial_sha256'] != sha256(trial_path) or m['seed'] != seed or m['arm'] != arm
                    or m['source_checkpoint_sha256'] != expected['model_sha256']
                    or m['provenance_sha256'] != expected['provenance_sha256']):
                raise ValueError('Export differs from training gate checkpoint')
            name = arm + str(seed)
            networks[name] = model
            identities[name] = dict(path=str(path), sha256=sha256(path), metadata_sha256=sha256(path.with_suffix('.json')))
    if not networks:
        raise ValueError('No qualified TRAIN arms')
    result_rows = {name: [] for name in ('baseline', *networks)}
    for (row, rgb), old in zip(frames, reference['results']['baseline']['frames'], strict=True):
        base = experimental(rgb, 'mser')
        for name in result_rows:
            start = time.perf_counter()
            proposed = base if name == 'baseline' else supplement(base, networks[name].predict(rgb))
            localizer_ms = (time.perf_counter() - start) * 1000 if name != 'baseline' else 0.
            if proposed[:len(base)] != base or len(proposed) > min(12, len(base) + 2):
                raise ValueError('Baseline proposals or crop budget changed')
            found = classify_candidates(predictor, rgb, proposed)
            detections = select_detections(found, 'split_merged', 2.42913421076193)
            result = dict(id=row['id'], source=row['source'], frame_sha256=row['sha256'],
                truth=row['provisional_truth'], detections=detections, extra_calls=len(proposed) - len(base),
                dense_host_ms=localizer_ms, counts=detection_counts(row['provisional_truth'], detections))
            if name == 'baseline':
                check_baseline(result, old)
            result_rows[name].append(result)
    results = {}
    for name, rows in result_rows.items():
        counts = {label: {key: sum(r['counts'][label][key] for r in rows) for key in ('tp', 'fp', 'fn')} for label in LABELS}
        results[name] = dict(frames=rows, metrics=metrics(counts))
    result = dict(results=results, models=identities, classifier_sha256=sha256(CLASSIFIER),
        reference_sha256=sha256(reference_path), trial_sha256=sha256(trial_path),
        queue_sha256=review['queue_sha256'], review_sha256=sha256(review_path),
        helpers_sha256={p: sha256(p) for p in ('scripts/balloon_grid_runtime.py', 'scripts/balloon_fine_grid.py',
            'scripts/audit_balloon_dense_proposals.py', 'scripts/pi_balloon_search_bench.py',
            'scripts/evaluate_red_blue_development.py', 'scripts/research_predictor.py')},
        script_sha256=sha256(__file__), classifier_threshold=.8, matching_iou=.5, proposal_threshold=.1,
        selection='split_merged', split_max_aspect=2.42913421076193,
        scope='Repeatedly inspected development panels, not independent qualification. Same crop classifier and budget; localizers never accept objects directly.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print({k: v['metrics'] for k, v in results.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=PANELS, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluate(args.panel, args.output)
