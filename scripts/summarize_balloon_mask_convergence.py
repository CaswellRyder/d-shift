"""Audit every paired classifier/search result; keep TRAIN, development and Pi separate."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_balloon_mask_convergence import classifiers
from scripts.export_balloon_mask_convergence import TRIAL
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_balloon_grid_clutter import nonregression


def meets_target(metrics):
    return set(metrics) == {'red_balloon', 'blue_balloon'} and all(
        m['precision'] is not None and m['recall'] is not None
        and m['precision'] >= .94 and m['recall'] >= .94 for m in metrics.values())


def summarize(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _, identities = classifiers()
    searches = ('baseline', 'coarse47', 'coarse48')
    names = {f'{m}/{s}' for m in identities for s in searches}
    panels, comparisons = {}, []
    for panel in PANELS:
        path = Path(f'runs/balloon-mask-convergence-scenes-{panel}-20261011.json')
        doc = read_json(path)
        reference_path = Path(f'runs/balloon-grid-convergence-scenes-{panel}-20261011.json')
        reference = read_json(reference_path)
        review_path = f'configs/{panel}.json'
        review = read_json(review_path)
        if (doc['classifiers'] != identities or set(doc['results']) != names
                or doc['classifier_threshold'] != .8 or doc['matching_iou'] != .5 or doc['proposal_threshold'] != .1
                or doc['selection'] != 'split_merged' or doc['split_max_aspect'] != 2.42913421076193
                or doc['training_trial_sha256'] != sha256(TRIAL)
                or doc['reference_sha256'] != sha256(reference_path)
                or doc['review_sha256'] != sha256(review_path) or doc['queue_sha256'] != review['queue_sha256']
                or doc['script_sha256'] != sha256('scripts/evaluate_balloon_mask_convergence.py')
                or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())
                or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
            raise ValueError('Evaluation scope or identity changed')
        for search in searches:
            for a, b in zip(doc['results'][f'parent/{search}']['frames'], reference['results'][search]['frames'], strict=True):
                check_baseline(a, b)
            if search != 'baseline' and doc['searches'][search] != reference['models'][search]:
                raise ValueError('Search identity changed')
        for result in doc['results'].values():
            for a, b in zip(result['frames'], doc['results']['parent/baseline']['frames'], strict=True):
                if any(a[k] != b[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Different frames or labels in paired evaluation')
        results = {name: checked_metrics(result) for name, result in doc['results'].items()}
        for seed in (47, 48):
            for search in searches:
                new = results[f'all_correct{seed}/{search}']
                comparisons.append(dict(panel=panel, seed=seed, search=search,
                    nonregression_vs_parent=nonregression(new, results[f'parent/{search}']),
                    nonregression_vs_accepted_only=nonregression(new, results[f'accepted{seed}/{search}']),
                    meets_target=meets_target(new)))
        panels[panel] = dict(path=str(path), sha256=sha256(path), frames=len(doc['results']['parent/baseline']['frames']), results=results)
    targets = {name: all(meets_target(p['results'][name]) for p in panels.values()) for name in sorted(names)}
    exports = {}
    for name, identity in identities.items():
        if name != 'parent':
            m = read_json(Path(identity['path']).with_suffix('.json'))
            exports[name] = dict(identity, bytes=m['bytes'], parity_rows=m['parity_rows'], max_delta=m['max_keras_tflite_delta'])
    result = dict(training_trial_sha256=sha256(TRIAL), exports=exports, panels=panels,
        comparisons=comparisons, per_variant_all_panels_target=targets,
        decision=('Development target reached by at least one fixed variant; independent qualification and Pi testing still missing.' if any(targets.values()) else
                  'Do not promote: TRAIN fit improved but no fixed variant reaches94% per-color precision and recall across the development panels.'),
        scope='Repeated related development imagery, not independent qualification. Training convergence and export parity do not establish generalization or Pi readiness.',
        script_sha256=sha256(__file__), test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(dict(targets=targets, decision=result['decision']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    summarize(parser.parse_args().output)
