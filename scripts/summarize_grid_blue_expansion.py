"""Audit paired scene outputs; no promotion based on sparse TRAIN coverage."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_grid_blue_expansion import models, CLASSIFIER, CLASSIFIER_SHA
from scripts.export_grid_blue_expansion import TRIAL
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_dense_background_trial import check_baseline
from scripts.summarize_balloon_grid_clutter import nonregression
from scripts.summarize_balloon_mask_convergence import meets_target

OUTPUT = Path('research/balloon-red-blue-20261008/grid-blue-expansion-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    _, identities = models()
    classifier = dict(path=str(CLASSIFIER), sha256=sha256(CLASSIFIER), metadata_sha256=sha256(CLASSIFIER.with_suffix('.json')))
    if classifier['sha256'] != CLASSIFIER_SHA:
        raise ValueError('Classifier changed')
    panels, comparisons = {}, []
    for panel in PANELS:
        path = Path(f'runs/grid-blue-expansion-scenes-{panel}-20261011.json')
        doc = read_json(path)
        reference_path = Path(f'runs/real-blue-clutter-scenes-{panel}-20261011.json')
        reference = read_json(reference_path)
        review_path = f'configs/{panel}.json'
        review = read_json(review_path)
        if (doc['models'] != identities or doc['classifier'] != classifier
                or set(doc['results']) != {'baseline', *identities}
                or doc['trial_sha256'] != sha256(TRIAL) or doc['reference_sha256'] != sha256(reference_path)
                or doc['review_sha256'] != sha256(review_path) or doc['queue_sha256'] != review['queue_sha256']
                or doc['classifier_threshold'] != .8 or doc['matching_iou'] != .5 or doc['proposal_threshold'] != .1
                or doc['selection'] != 'split_merged' or doc['split_max_aspect'] != 2.42913421076193
                or doc['script_sha256'] != sha256('scripts/evaluate_grid_blue_expansion.py')
                or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())
                or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
            raise ValueError('Scene evaluation provenance changed')
        for new, old in (('baseline', 'admit48/baseline'), ('parent', 'admit48/coarse47')):
            for a, b in zip(doc['results'][new]['frames'], reference['results'][old]['frames'], strict=True):
                check_baseline(a, b)
        for result in doc['results'].values():
            for a, b in zip(result['frames'], doc['results']['baseline']['frames'], strict=True):
                if any(a[k] != b[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Paired frame truth changed')
        results = {name: checked_metrics(result) for name, result in doc['results'].items()}
        for seed in (47, 48):
            new = results[f'admit{seed}']
            comparisons.append(dict(panel=panel, seed=seed,
                nonregression_vs_parent=nonregression(new, results['parent']),
                nonregression_vs_control=nonregression(new, results[f'control{seed}']), meets_target=meets_target(new)))
        panels[panel] = dict(path=str(path), sha256=sha256(path), frames=len(doc['results']['baseline']['frames']), results=results)
    targets = {name: all(meets_target(p['results'][name]) for p in panels.values()) for name in ('baseline', *identities)}
    exports = {}
    for name, identity in identities.items():
        if name == 'parent':
            continue
        meta = read_json(Path(identity['path']).with_suffix('.json'))
        exports[name] = dict(identity, bytes=meta['bytes'], parameters=meta['parameters'],
            decoded_parity_frames=meta['decoded_parity_frames'], max_tensor_delta=meta['max_tensor_delta'])
    decision = ('Development target reached; independent qualification and Pi checks remain.' if any(targets.values()) else
                'Do not promote: no fixed variant reaches94% precision and recall for both colors across development panels.')
    write_json(OUTPUT, dict(trial_sha256=sha256(TRIAL), classifier=classifier, exports=exports, panels=panels,
        comparisons=comparisons, per_variant_all_panels_target=targets, decision=decision,
        script_sha256=sha256(__file__), scope='Repeated related development only; not independent qualification or Pi timing.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(targets=targets, comparisons=comparisons, decision=decision))


if __name__ == '__main__':
    summarize()
