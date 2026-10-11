"""Verify the full-scene decision; crop TRAIN retention is not generalization."""
from pathlib import Path

from dtr.data import read_json,sha256,write_json
from scripts.evaluate_blue_objects import OUTPUT as DEVELOPMENT,RAW
from scripts.export_blue_objects import checked_trial,TRIAL
from scripts.summarize_public_balloon_trial import checked_metrics
from scripts.summarize_balloon_grid_clutter import nonregression
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.audit_current_blue_search import matched_truth

OUTPUT = Path('research/balloon-red-blue-20261008/blue-object-trial-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial = checked_trial()
    doc,raw = read_json(DEVELOPMENT),read_json(RAW)
    if (doc['training_trial_sha256']!=sha256(TRIAL) or doc['raw']['sha256']!=sha256(RAW)
            or doc['script_sha256']!=sha256('scripts/evaluate_blue_objects.py')
            or any(sha256(p)!=h for p,h in doc['helpers_sha256'].items())
            or any(doc[k] is not False for k in ('test_evaluated','pi_tested','deployment_approved','qualification_achieved'))):
        raise ValueError('Evaluation identity/scope changed')
    for identity in doc['classifiers'].values():
        path = Path(identity['path'])
        if sha256(path)!=identity['sha256'] or sha256(path.with_suffix('.json'))!=identity['metadata_sha256']:
            raise ValueError('Classifier changed')
    comparisons = []
    for panel,p in doc['panels'].items():
        values = {k:checked_metrics(r) for k,r in raw[panel].items()}
        if values!=p['results']:
            raise ValueError('Recomputed metrics changed')
        for seed in (47,48):
            for search in ('baseline','coarse47'):
                key = f'admit{seed}/{search}'
                losses = []
                for r,base in zip(raw[panel][key]['frames'],raw[panel][f'parent/{search}']['frames'],strict=True):
                    if any(r[k]!=base[k] for k in ('source','frame_sha256','truth')):
                        raise ValueError('Different comparison frames')
                    losses.extend(dict(source=r['source'],target=r['truth'][i]) for i in sorted(matched_truth(base)-matched_truth(r)))
                comparisons.append(dict(panel=panel,seed=seed,search=search,
                    nonregression_vs_parent=nonregression(values[key],values[f'parent/{search}']),
                    nonregression_vs_control=nonregression(values[key],values[f'control{seed}/{search}']),
                    lost_parent_targets=losses,meets_target=meets_target(values[key])))
    if comparisons!=doc['comparisons']:
        raise ValueError('Comparison receipt differs')
    variants = {f'admit{seed}/{search}':dict(
        nonregression_all_panels=all(r['nonregression_vs_parent'] and r['nonregression_vs_control']
            and not r['lost_parent_targets'] for r in comparisons if r['seed']==seed and r['search']==search),
        meets_target_all_panels=all(r['meets_target'] for r in comparisons if r['seed']==seed and r['search']==search))
        for seed in (47,48) for search in ('baseline','coarse47')}
    if any(r['nonregression_all_panels'] for r in variants.values()):
        raise ValueError('Revisit rejection: unexpected nonregressing variant')
    write_json(OUTPUT,dict(variants=variants,paired_training_passed=trial['eligible_for_development'],
        training_receipt_sha256=sha256(TRIAL),development_receipt_sha256=sha256(DEVELOPMENT),
        script_sha256=sha256(__file__),
        decision='Reject both admitted checkpoints for deployment: training retention did not generalize. Preserve prior research and Pi incumbents.',
        admission_receipt_sha256=sha256('research/balloon-red-blue-20261008/blue-object-admission-20261011.json'),
        loss_audit_sha256=sha256('research/balloon-red-blue-20261008/blue-object-losses-20261011.json'),
        next_action='Protect and expand real TRAIN balloon crop-view coverage while retaining learned clutter rejection; no IMG development mining or confidence-threshold tuning. Separate localization remains unresolved.',
        exports={name:read_json(Path(identity['path']).with_suffix('.json')) for name,identity in doc['classifiers'].items() if name!='parent'},
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(variants)


if __name__=='__main__':
    summarize()


