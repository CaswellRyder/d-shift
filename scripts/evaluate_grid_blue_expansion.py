"""Fixed classifier, thresholds and crop budget for paired proposal evaluation."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.balloon_dense_runtime import DenseProposals
from scripts.balloon_grid_runtime import GridProposals
from scripts.audit_balloon_dense_proposals import supplement
from scripts.export_grid_blue_expansion import checked_trial, TRIAL
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_dense_background_trial import check_baseline

CLASSIFIER = Path('runs/real-blue-clutter-convergence-admit-seed48-20261011/student.fp32.tflite')
CLASSIFIER_SHA = '62ecb93da3d3e6f94ca0e1213520fa04191d0ad5666810e489ba0ed26d4ff3c1'


def models():
    trial = checked_trial()
    parent = Path('runs/balloon-grid-convergence-coarse-seed47-20261011/proposal.fp32.tflite')
    paths = dict(parent=parent)
    paths.update({name: Path(row['path']) / 'proposal.fp32.tflite' for name, row in trial['trials'].items()})
    networks, identities = {}, {}
    for name, path in paths.items():
        if name == 'parent':
            model = GridProposals(path)
        else:
            model = DenseProposals(path)
            m, r = model.metadata, trial['trials'][name]['report']
            if (m['experiment'] != 'grid_blue_expansion' or m['trial_sha256'] != sha256(TRIAL)
                    or m['mode'] + str(m['seed']) != name or m['source_checkpoint_sha256'] != r['model_sha256']
                    or m['provenance_sha256'] != r['provenance_sha256']
                    or m['export_script_sha256'] != sha256('scripts/export_grid_blue_expansion.py')
                    or m['decoded_parity_frames'] != 156 or not 0 <= m['max_tensor_delta'] <= .002
                    or m['parameters'] != 2342 or m['pi_tested'] is not False):
                raise ValueError('Export differs from gated trial')
        networks[name] = model
        identities[name] = dict(path=str(path), sha256=sha256(path), metadata_sha256=sha256(path.with_suffix('.json')))
    return networks, identities


def evaluate(panel):
    output = Path(f'runs/grid-blue-expansion-scenes-{panel}-20261011.json')
    if output.exists():
        raise FileExistsError(output)
    reference_path = f'runs/real-blue-clutter-scenes-{panel}-20261011.json'
    reference = read_json(reference_path)
    frames, review = reviewed_frames('data/' + panel, f'configs/{panel}.json')
    networks, identities = models()
    classifier = dict(path=str(CLASSIFIER), sha256=sha256(CLASSIFIER), metadata_sha256=sha256(CLASSIFIER.with_suffix('.json')))
    if (classifier['sha256'] != CLASSIFIER_SHA or classifier != reference['classifiers']['admit48']
            or identities['parent'] != reference['searches']['coarse47']
            or review['threshold'] != .8 or review['matching_iou'] != .5
            or review['queue_sha256'] != reference['queue_sha256']
            or sha256(f'configs/{panel}.json') != reference['review_sha256']):
        raise ValueError('Fixed comparison identity changed')
    predictor = Predictor(CLASSIFIER, allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Classifier threshold changed')
    rows = {name: [] for name in ('baseline', *networks)}
    for index, (row, rgb) in enumerate(frames):
        base = experimental(rgb, 'mser')
        for name in rows:
            proposed = base if name == 'baseline' else supplement(base, networks[name].predict(rgb))
            if proposed[:len(base)] != base or len(proposed) > min(12, len(base) + 2):
                raise ValueError('Search budget changed')
            found = classify_candidates(predictor, rgb, proposed)
            detections = select_detections(found, 'split_merged', 2.42913421076193)
            result = dict(id=row['id'], source=row['source'], frame_sha256=row['sha256'],
                truth=row['provisional_truth'], detections=detections, extra_calls=len(proposed) - len(base),
                counts=detection_counts(row['provisional_truth'], detections))
            if name in ('baseline', 'parent'):
                key = 'baseline' if name == 'baseline' else 'coarse47'
                check_baseline(result, reference['results'][f'admit48/{key}']['frames'][index])
            rows[name].append(result)
        if (index + 1) % 20 == 0:
            print(dict(panel=panel, frames=index + 1), flush=True)
    results = {}
    for name, subset in rows.items():
        counts = {label: {key: sum(row['counts'][label][key] for row in subset) for key in ('tp', 'fp', 'fn')} for label in LABELS}
        results[name] = dict(frames=subset, metrics=metrics(counts))
    write_json(output, dict(results=results, models=identities, classifier=classifier,
        reference_sha256=sha256(reference_path), trial_sha256=sha256(TRIAL),
        queue_sha256=review['queue_sha256'], review_sha256=sha256(f'configs/{panel}.json'),
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/balloon_dense_runtime.py', 'scripts/balloon_grid_runtime.py',
            'scripts/audit_balloon_dense_proposals.py', 'scripts/research_predictor.py',
            'scripts/research_balloon_search.py', 'scripts/pi_balloon_search_bench.py',
            'scripts/evaluate_red_blue_development.py')},
        classifier_threshold=.8, matching_iou=.5, proposal_threshold=.1,
        selection='split_merged', split_max_aspect=2.42913421076193,
        scope='Repeated related development, not qualification. Same fixed classifier and bounded MSER-plus2 search.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print({name: row['metrics'] for name, row in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=PANELS, required=True)
    evaluate(parser.parse_args().panel)
