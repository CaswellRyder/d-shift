"""Fixed development ablation of the coarse blue proposal cap; never a Pi default."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_grid_runtime import GridProposals
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

OUTPUT = Path('research/balloon-red-blue-20261008/coarse-blue-budget-20261011.json')
RAW = Path('runs/coarse-blue-budget-20261011.json')
BUDGETS = (2, 4, 6)


def supplement_budget(base, learned, budget):
    if type(budget) is not int or budget not in BUDGETS or len(base) > 12:
        raise ValueError('Undeclared budget or oversized base')
    found = list(base)
    for row in learned:
        if len(found) >= min(12, len(base) + budget):
            break
        if row['color_group'] == 1 and not any(iou(row['box'], r['box']) > .65 for r in found):
            found.append(row)
    return found


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve frozen diagnostic')
    if sha256(MODEL) != MODEL_SHA or sha256(SEARCH) != SEARCH_SHA:
        raise ValueError('Frozen classifier or localizer changed')
    predictor, grid = Predictor(MODEL, allow_unvalidated=True), GridProposals(SEARCH)
    panels, raw = {}, {}
    for panel in PANELS:
        reference_path = Path(f'runs/real-blue-clutter-scenes-{panel}-20261011.json')
        reference = read_json(reference_path)
        sources, review = reviewed_frames('data/' + panel, 'configs/' + panel + '.json')
        if (review['threshold'] != .8 or review['matching_iou'] != .5
                or reference['review_sha256'] != sha256('configs/' + panel + '.json')
                or reference['queue_sha256'] != review['queue_sha256']
                or reference['classifiers']['admit48']['sha256'] != MODEL_SHA
                or reference['searches']['coarse47']['sha256'] != SEARCH_SHA):
            raise ValueError('Frozen reference contract changed')
        variants = {str(n): [] for n in BUDGETS}
        for i, (r, rgb) in enumerate(sources):
            base, learned = experimental(rgb, 'mser'), grid.predict(rgb)
            for n in BUDGETS:
                candidates = supplement_budget(base, learned, n)
                ds = select_detections(classify_candidates(predictor, rgb, candidates), 'split_merged', 2.42913421076193)
                row = dict(id=r['id'], source=r['source'], frame_sha256=r['sha256'], truth=r['provisional_truth'],
                    detections=ds, counts=detection_counts(r['provisional_truth'], ds),
                    extra_calls=len(candidates) - len(base), neural_calls=len(candidates))
                if n == 2:
                    check_baseline(row, reference['results']['admit48/coarse47']['frames'][i])
                variants[str(n)].append(row)
        results = {}
        raw[panel] = {}
        for key, frames in variants.items():
            counts = {c: {k: sum(f['counts'][c][k] for f in frames) for k in ('tp', 'fp', 'fn')} for c in LABELS}
            values = metrics(counts)
            raw[panel][key] = dict(frames=frames, metrics=values)
            results[key] = dict(metrics=values, **compare_frames(frames, variants['2']),
                neural_calls=sum(f['neural_calls'] for f in frames), max_neural_calls=max(f['neural_calls'] for f in frames),
                extra_calls=sum(f['extra_calls'] for f in frames), meets_target=meets_target(values))
        panels[panel] = dict(frames=len(sources), results=results, reference_sha256=sha256(reference_path),
            queue_sha256=review['queue_sha256'], review_sha256=sha256('configs/' + panel + '.json'))
        print(panel, {k: dict(blue=v['metrics']['blue_balloon'], calls=v['neural_calls'],
            oracle=v['colors']['blue_balloon']['geometric_oracle_covered']) for k, v in results.items()}, flush=True)
    write_json(RAW, raw)
    write_json(OUTPUT, dict(panels=panels, budgets=BUDGETS, max_total_proposals=12,
        classifier=dict(path=str(MODEL), sha256=MODEL_SHA, metadata_sha256=sha256(MODEL.with_suffix('.json'))),
        localizer=dict(path=str(SEARCH), sha256=SEARCH_SHA, metadata_sha256=sha256(SEARCH.with_suffix('.json'))),
        raw=dict(path=str(RAW), sha256=sha256(RAW)), script_sha256=sha256(__file__),
        helpers_sha256={p: sha256(p) for p in ('scripts/research_balloon_search.py',
            'scripts/balloon_grid_runtime.py', 'scripts/research_predictor.py', 'scripts/pi_balloon_search_bench.py',
            'scripts/audit_indoor5_outcomes.py', 'scripts/audit_balloon_pipeline_errors.py')},
        scope='Fixed two/four/six blue-proposal ablation on repeated development panels. Calls are workload counts, not measured Pi latency. Oracle uses truth, not deployable accuracy.',
        classifier_threshold=.8, matching_iou=.5, training_performed=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
