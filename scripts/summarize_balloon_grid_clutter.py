"""Audit paired clutter results and apply the predeclared nonregression gate."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_balloon_grid_clutter import checked_models
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_dense_background_trial import check_baseline
from scripts.train_balloon_grid_clutter import PROTOCOL


def nonregression(candidate, reference):
    if set(candidate) != {'red_balloon', 'blue_balloon'} or set(reference) != set(candidate):
        raise ValueError('Both colors required')
    return all(candidate[c]['tp'] >= reference[c]['tp'] and candidate[c]['fp'] <= reference[c]['fp']
               and candidate[c]['tp'] + candidate[c]['fn'] == reference[c]['tp'] + reference[c]['fn']
               for c in candidate)


def summarize(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _, identities = checked_models()
    searches = ('baseline', 'coarse47', 'coarse48')
    names = {f'{m}/{s}' for m in identities for s in searches}
    panels, gates = {}, []
    for panel in PANELS:
        path = Path(f'runs/balloon-grid-clutter-scenes-{panel}-20261011.json')
        doc = read_json(path)
        reference_path = Path(f'runs/balloon-grid-convergence-scenes-{panel}-20261011.json')
        reference = read_json(reference_path)
        review_path = f'configs/{panel}.json'
        review = read_json(review_path)
        if (doc['classifiers'] != identities or set(doc['results']) != names
                or doc['classifier_threshold'] != .8 or doc['matching_iou'] != .5
                or doc['proposal_threshold'] != .1 or doc['selection'] != 'split_merged'
                or doc['split_max_aspect'] != 2.42913421076193
                or doc['protocol_sha256'] != sha256(PROTOCOL)
                or doc['reference_sha256'] != sha256(reference_path)
                or doc['review_sha256'] != sha256(review_path)
                or doc['queue_sha256'] != review['queue_sha256']
                or doc['script_sha256'] != sha256('scripts/evaluate_balloon_grid_clutter.py')
                or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())
                or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
            raise ValueError('Evaluation identity or scope changed')
        for search in searches:
            for a, b in zip(doc['results'][f'parent/{search}']['frames'], reference['results'][search]['frames'], strict=True):
                check_baseline(a, b)
        for name, result in doc['results'].items():
            for a, b in zip(result['frames'], doc['results']['parent/baseline']['frames'], strict=True):
                if any(a[k] != b[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Classifiers evaluated on different frames or labels')
        results = {name: checked_metrics(result) for name, result in doc['results'].items()}
        for seed in (47, 48):
            for search in searches:
                candidate = results[f'admit{seed}/{search}']
                gate = dict(panel=panel, seed=seed, search=search,
                    versus_parent=nonregression(candidate, results[f'parent/{search}']),
                    versus_control=nonregression(candidate, results[f'control{seed}/{search}']))
                gates.append(gate)
        panels[panel] = dict(path=str(path), sha256=sha256(path), frames=len(doc['results']['parent/baseline']['frames']), results=results)
    training = {}
    for name, identity in identities.items():
        if name == 'parent':
            continue
        run = Path(identity['path']).parent
        report = read_json(run / 'report.json')
        training[name] = dict(ordinary_before=report['ordinary_before'], ordinary_after=report['ordinary_after'],
            reviewed_probe={view: {point: sum(r[f'{point}_prediction'] == 'background' for r in report['probe'] if r['view'] == view)
                for point in ('before', 'after')} for view in ('original', 'blue_surface')},
            mean_synthetic_changed_fraction=report['mean_synthetic_changed_fraction'],
            parity_rows=report['parity_rows'], max_keras_tflite_delta=report['max_keras_tflite_delta'],
            report_sha256=sha256(run / 'report.json'), bytes=Path(identity['path']).stat().st_size)
    passed = all(g['versus_parent'] and g['versus_control'] for g in gates)
    target = {name: all((v['precision'] or 0) >= .94 and (v['recall'] or 0) >= .94
        for panel in panels.values() for key, result in panel['results'].items() if key.startswith(name + '/')
        for v in result.values()) for name in identities if name != 'parent'}
    result = dict(protocol_sha256=sha256(PROTOCOL), classifiers=identities, training=training, panels=panels,
        gates=gates, development_nonregression_pass=passed, all_searches_panels_target=target,
        decision=('Research nonregression gate passed; independent qualification still missing.' if passed else
                  'Reject runtime promotion: background rejection improves precision but loses real red and blue detections.'),
        next_step='Keep deployed model unchanged. Investigate TRAIN-only class-preserving adaptation or separately gated blue verification; do not mine development crops or lower qualification standards.',
        scope='Fixed paired final-epoch TRAIN experiment and repeated related development panels. Synthetic views are not new real samples. No pooling, live Pi timing or qualification claim.',
        script_sha256=sha256(__file__), test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(output, result)
    print(dict(development_nonregression_pass=passed, target=target, decision=result['decision']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    summarize(parser.parse_args().output)
