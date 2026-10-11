"""Recompute paired scene metrics and distinguish development from qualification."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_balloon_blue_final import classifiers
from scripts.export_balloon_blue_final import TRIAL
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_balloon_grid_clutter import nonregression
from scripts.summarize_balloon_mask_convergence import meets_target

OUTPUT = Path('research/balloon-red-blue-20261008/blue-final-fit-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    _, identities = classifiers()
    searches = ('baseline', 'coarse47', 'coarse48')
    names = {f'{m}/{s}' for m in identities for s in searches}
    panels, comparisons = {}, []
    for panel in PANELS:
        path = Path(f'runs/balloon-blue-final-scenes-{panel}-20261011.json')
        doc = read_json(path)
        reference_path = Path(f'runs/balloon-mask-convergence-scenes-{panel}-20261011.json')
        reference = read_json(reference_path)
        review_path = f'configs/{panel}.json'
        review = read_json(review_path)
        if (doc['classifiers'] != identities or set(doc['results']) != names
                or doc['classifier_threshold'] != .8 or doc['matching_iou'] != .5 or doc['proposal_threshold'] != .1
                or doc['selection'] != 'split_merged' or doc['split_max_aspect'] != 2.42913421076193
                or doc['training_trial_sha256'] != sha256(TRIAL)
                or doc['reference_sha256'] != sha256(reference_path)
                or doc['review_sha256'] != sha256(review_path) or doc['queue_sha256'] != review['queue_sha256']
                or doc['script_sha256'] != sha256('scripts/evaluate_balloon_blue_final.py')
                or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())
                or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
            raise ValueError('Evaluation identity or scope changed')
        for search in searches:
            for a, b in zip(doc['results'][f'parent/{search}']['frames'], reference['results'][f'accepted47/{search}']['frames'], strict=True):
                check_baseline(a, b)
            if search != 'baseline' and doc['searches'][search] != reference['searches'][search]:
                raise ValueError('Search identity changed')
        for result in doc['results'].values():
            for a, b in zip(result['frames'], doc['results']['parent/baseline']['frames'], strict=True):
                if any(a[k] != b[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Different frame labels in paired evaluation')
        results = {name: checked_metrics(result) for name, result in doc['results'].items()}
        for seed in (47, 48):
            for search in searches:
                new = results[f'admit{seed}/{search}']
                comparisons.append(dict(panel=panel, seed=seed, search=search,
                    nonregression_vs_parent=nonregression(new, results[f'parent/{search}']),
                    nonregression_vs_control=nonregression(new, results[f'control{seed}/{search}']),
                    meets_target=meets_target(new)))
        panels[panel] = dict(path=str(path), sha256=sha256(path),
            frames=len(doc['results']['parent/baseline']['frames']), results=results)
    targets = {name: all(meets_target(p['results'][name]) for p in panels.values()) for name in sorted(names)}
    exports = {}
    for name, identity in identities.items():
        if name != 'parent':
            meta = read_json(Path(identity['path']).with_suffix('.json'))
            exports[name] = dict(identity, bytes=meta['bytes'], parity_rows=meta['parity_rows'],
                max_delta=meta['max_keras_tflite_delta'], parameters=meta['parameters'])
    decision = ('Development target reached by a fixed variant; independent qualification and Pi tests still required.' if any(targets.values()) else
                'Do not promote: no fixed variant reaches94% precision and recall for both colors across the development panels.')
    write_json(OUTPUT, dict(training_trial_sha256=sha256(TRIAL), exports=exports,
        panels=panels, comparisons=comparisons, per_variant_all_panels_target=targets,
        decision=decision, scope='Repeated related development frames; not independent final qualification. No accuracy pooling or live timing claims.',
        script_sha256=sha256(__file__), test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(targets=targets, decision=decision))


if __name__ == '__main__':
    summarize()
