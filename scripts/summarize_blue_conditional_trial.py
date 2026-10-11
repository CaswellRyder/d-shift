"""Independently recount the crop-view experiment; development never qualifies flight."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.evaluate_blue_conditional_trial import OUTPUT as DEVELOPMENT, RAW
from scripts.export_blue_conditional_trial import checked_trial, TRIAL
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_balloon_grid_clutter import nonregression
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_balloon_fragments import PANELS
from scripts.audit_current_blue_search import matched_truth

OUTPUT = Path('research/balloon-red-blue-20261008/blue-conditional-trial-decision-20261011.json')
SEARCHES = ('baseline', 'coarse47')
SEEDS = (47, 48)


def variant_gates(comparisons):
    expected = {(p, s, m) for p in PANELS for s in SEEDS for m in SEARCHES}
    keys = [(r['panel'], r['seed'], r['search']) for r in comparisons]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Require each declared panel, seed and search exactly once')
    return {f'admit{seed}/{search}': dict(
        nonregression_all_panels=all(r['nonregression_vs_parent'] and r['nonregression_vs_control']
            and not r['lost_parent_targets'] for r in comparisons if r['seed'] == seed and r['search'] == search),
        meets_target_all_panels=all(r['meets_target'] for r in comparisons
            if r['seed'] == seed and r['search'] == search))
        for seed in SEEDS for search in SEARCHES}


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial = checked_trial()
    doc, raw = read_json(DEVELOPMENT), read_json(RAW)
    if (doc['training_trial_sha256'] != sha256(TRIAL) or doc['raw']['sha256'] != sha256(RAW)
            or doc['script_sha256'] != sha256('scripts/evaluate_blue_conditional_trial.py')
            or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())
            or set(doc['panels']) != set(PANELS) or set(raw) != set(PANELS)
            or any(doc[k] is not False for k in ('test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Evaluation identity/scope changed')
    for identity in doc['classifiers'].values():
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Classifier changed')
    comparisons = []
    for panel, p in doc['panels'].items():
        review_path = Path(f'configs/{panel}.json')
        reference = Path(f'runs/real-blue-clutter-scenes-{panel}-20261011.json')
        search = Path(p['search']['path'])
        if (sha256(review_path) != p['review_sha256'] or read_json(review_path)['queue_sha256'] != p['queue_sha256']
                or sha256(reference) != p['reference_sha256'] or sha256(search) != p['search']['sha256']
                or sha256(search.with_suffix('.json')) != p['search']['metadata_sha256']):
            raise ValueError('Panel or search identity changed')
        values = {k: checked_metrics(r) for k, r in raw[panel].items()}
        if values != p['results']:
            raise ValueError('Recomputed metrics changed')
        for result in raw[panel].values():
            if len(result['frames']) != p['frames']:
                raise ValueError('Incomplete comparison')
            for r, base in zip(result['frames'], raw[panel]['parent/baseline']['frames'], strict=True):
                if any(r[k] != base[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Different comparison frames')
        for seed in SEEDS:
            for search_name in SEARCHES:
                key = f'admit{seed}/{search_name}'
                losses = []
                for r, base in zip(raw[panel][key]['frames'], raw[panel][f'parent/{search_name}']['frames'], strict=True):
                    losses.extend(dict(source=r['source'], target=r['truth'][i])
                        for i in sorted(matched_truth(base) - matched_truth(r)))
                comparisons.append(dict(panel=panel, seed=seed, search=search_name,
                    nonregression_vs_parent=nonregression(values[key], values[f'parent/{search_name}']),
                    nonregression_vs_control=nonregression(values[key], values[f'control{seed}/{search_name}']),
                    lost_parent_targets=losses, meets_target=meets_target(values[key])))
    # JSON key sorting can change panel iteration order; compare by explicit identity.
    def key(r):
        return r['panel'], r['seed'], r['search']
    if sorted(comparisons, key=key) != sorted(doc['comparisons'], key=key):
        raise ValueError('Comparison receipt differs')
    variants = variant_gates(comparisons)
    eligible = [k for k, v in variants.items() if v['nonregression_all_panels']]
    write_json(OUTPUT, dict(variants=variants, paired_training_passed=trial['eligible_for_development'],
        training_receipt_sha256=sha256(TRIAL), development_receipt_sha256=sha256(DEVELOPMENT),
        script_sha256=sha256(__file__),
        decision=('Research nonregression passed; independent qualification and live Pi checks still required.' if eligible else
                  'Reject runtime promotion: reviewed TRAIN crop retention did not generalize without regressions.'),
        research_nonregressing_variants=eligible,
        intervention='Shared1764-row admitted curriculum plus11 new reviewed blue crop views repeated twice versus11 old blue controls repeated twice. Same9844parameter conditional model with2081trainable parameters; exact frozen7763parameter parent and preserved red probabilities.400epochs/44800steps, fresh Adam cosine1e-4 to1e-5; final checkpoints only.',
        scope='Repeated related development panels, not independent test or live-camera qualification. No deployment change.',
        exports={name: read_json(Path(identity['path']).with_suffix('.json'))
                 for name, identity in doc['classifiers'].items() if name != 'parent'},
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(variants)


if __name__ == '__main__':
    summarize()
