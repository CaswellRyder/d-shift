"""Fixed paired full-scene evaluation; no repeated-panel qualification claims."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import supplement
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.balloon_grid_runtime import GridProposals
from scripts.blue_scene_localizer_data import PROTOCOL, run_path
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.pi_balloon_search_bench import select_detections
from scripts.prepare_indoor5_runtime_review import SEARCH, SEARCH_SHA
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.summarize_balloon_fragments import PANELS
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_dense_background_trial import check_baseline


def nonregression(parent, candidate):
    return all(candidate[c]['tp'] >= parent[c]['tp'] and candidate[c]['fp'] <= parent[c]['fp'] for c in LABELS)


def evaluate():
    output = Path('research/balloon-red-blue-20261008/blue-scene-localizer-development-20261011.json')
    raw_path = Path('runs/blue-scene-localizer-development-20261011.json')
    training = Path('research/balloon-red-blue-20261008/blue-scene-localizer-training-20261011.json')
    if output.exists() or raw_path.exists():
        raise FileExistsError('Preserve fixed comparison')
    receipt = read_json(training)
    if (sha256(MODEL) != MODEL_SHA or sha256(SEARCH) != SEARCH_SHA
            or receipt['protocol_sha256'] != sha256(PROTOCOL)
            or receipt['script_sha256'] != sha256('scripts/export_blue_scene_localizer.py')):
        raise ValueError('Frozen classifier, localizer or audit changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    if predictor.metadata['threshold'] != .8:
        raise ValueError('Classifier operating point changed')
    networks = {'parent': GridProposals(SEARCH)}
    identities = {}
    for arm in ('control', 'background'):
        for seed in (47, 48):
            key = f'{arm}{seed}'
            run = run_path(arm, seed)
            path = run/'proposal.fp32.tflite'
            model = DenseProposals(path)
            if (model.metadata != receipt['results'][key]['metadata']
                    or model.metadata['provenance_sha256'] != sha256(run/'provenance.json')
                    or model.metadata['source_checkpoint_sha256'] != sha256(run/'proposal.keras')):
                raise ValueError('Export does not match audited checkpoint')
            networks[key] = model
            identities[key] = dict(path=str(path), sha256=sha256(path), metadata_sha256=sha256(path.with_suffix('.json')))
    raw, panels = {}, {}
    for panel in PANELS:
        ref_path = Path(f'runs/real-blue-clutter-scenes-{panel}-20261011.json')
        reference = read_json(ref_path)
        sources, review = reviewed_frames('data/'+panel, 'configs/'+panel+'.json')
        if (review['threshold'] != .8 or review['matching_iou'] != .5
                or reference['review_sha256'] != sha256('configs/'+panel+'.json')
                or reference['queue_sha256'] != review['queue_sha256']
                or reference['classifiers']['admit48']['sha256'] != MODEL_SHA
                or reference['searches']['coarse47']['sha256'] != SEARCH_SHA):
            raise ValueError('Development reference changed')
        frames = {key: [] for key in networks}
        for i, (r, rgb) in enumerate(sources):
            base = experimental(rgb, 'mser')
            for key, network in networks.items():
                proposed = supplement(base, network.predict(rgb))
                if proposed[:len(base)] != base or len(proposed) > min(12, len(base)+2):
                    raise ValueError('Search budget or baseline changed')
                found = classify_candidates(predictor, rgb, proposed)
                ds = select_detections(found, 'split_merged', 2.42913421076193)
                row = dict(id=r['id'], source=r['source'], frame_sha256=r['sha256'], truth=r['provisional_truth'],
                    detections=ds, counts=detection_counts(r['provisional_truth'], ds),
                    neural_calls=len(proposed), extra_calls=len(proposed)-len(base))
                if key == 'parent':
                    check_baseline(row, reference['results']['admit48/coarse47']['frames'][i])
                frames[key].append(row)
        raw[panel], results = {}, {}
        for key, rows in frames.items():
            counts = {c: {k: sum(r['counts'][c][k] for r in rows) for k in ('tp', 'fp', 'fn')} for c in LABELS}
            values = metrics(counts)
            raw[panel][key] = dict(frames=rows, metrics=values)
            results[key] = dict(metrics=values, **compare_frames(rows, frames['parent']),
                neural_calls=sum(r['neural_calls'] for r in rows), meets_target=meets_target(values))
        for key in results:
            results[key]['nonregression'] = nonregression(results['parent']['metrics'], results[key]['metrics'])
        panels[panel] = dict(results=results, frames=len(sources), reference_sha256=sha256(ref_path),
            review_sha256=sha256('configs/'+panel+'.json'), queue_sha256=review['queue_sha256'])
        print(panel, {k: dict(metrics=v['metrics'], calls=v['neural_calls']) for k, v in results.items()}, flush=True)
    write_json(raw_path, raw)
    write_json(output, dict(panels=panels, models=identities, training_sha256=sha256(training),
        raw=dict(path=str(raw_path), sha256=sha256(raw_path)), script_sha256=sha256(__file__),
        classifier_sha256=MODEL_SHA, parent_localizer_sha256=SEARCH_SHA, protocol_sha256=sha256(PROTOCOL),
        helpers_sha256={p: sha256(p) for p in ('scripts/balloon_dense_runtime.py', 'scripts/balloon_grid_runtime.py',
            'scripts/audit_balloon_dense_proposals.py', 'scripts/research_balloon_search.py',
            'scripts/research_predictor.py', 'scripts/pi_balloon_search_bench.py',
            'scripts/evaluate_red_blue_development.py', 'scripts/audit_indoor5_outcomes.py')},
        scope='Repeated related development panels, not independent qualification. Same classifier, model size and search budget; call counts are not Pi latency.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    evaluate()
