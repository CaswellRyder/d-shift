"""One prespecified soft fusion experiment; no development/test scene access."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_balloon_hog_gate import TRAINING, unique_indices, verified_data
from scripts.blue_soft_shape_fusion import adjust, fit
from scripts.train_balloon_hog_gate import counts

PROTOCOL = Path('configs/blue-soft-shape-fusion-20261011.json')
RUN = Path('runs/blue-soft-shape-fusion-20261011')
OUTPUT = Path('research/balloon-red-blue-20261008/blue-soft-shape-fusion-20261011.json')


def accepted(p):
    return (p.argmax(axis=1)>0) & (p.max(axis=1)>=.8)


def improves(before, after):
    return bool(after['blue_balloon']['fp'] < before['blue_balloon']['fp']
        and after['blue_balloon']['f1'] > before['blue_balloon']['f1']
        and all(min(after[c]['precision'], after[c]['recall']) >= min(before[c]['precision'], before[c]['recall']) for c in before))


def run():
    if OUTPUT.exists() or RUN.exists():
        raise FileExistsError(RUN)
    protocol = read_json(PROTOCOL)
    if protocol['ridge_penalty'] != .1 or protocol['folds'] != 5:
        raise ValueError('Fixed recipe changed')
    _, rows, labels, pixels, x, groups, folds, saved = verified_data()
    p = saved['probabilities'].astype(np.float64)
    output, receipts = np.empty_like(p), []
    RUN.mkdir(parents=True)
    for fold in range(5):
        keep = folds != fold
        if set(np.asarray(groups)[keep]) & set(np.asarray(groups)[~keep]):
            raise ValueError('Source fold overlap')
        model = fit(x[keep], labels[keep], p[keep])
        output[~keep] = adjust(p[~keep], x[~keep], model)
        path = RUN/f'fold-{fold}.npz'
        np.savez_compressed(path, **model)
        receipts.append(dict(fold=fold, fitting_rows=int(keep.sum()), withheld_rows=int((~keep).sum()),
            model_sha256=sha256(path), iterations=int(model['iterations']), objective=float(model['objective'])))
    if not np.array_equal(p[:, 1], output[:, 1]) or not np.array_equal(p[p.argmax(1)==1], output[p.argmax(1)==1]):
        raise ValueError('Red confidence/parent red predictions changed')
    unique = unique_indices(rows, labels, pixels)
    old_accept, new_accept = accepted(p), accepted(output)
    before, after = counts(labels, p, old_accept), counts(labels, output, new_accept)
    unique_before = counts(labels[unique], p[unique], old_accept[unique])
    unique_after = counts(labels[unique], output[unique], new_accept[unique])
    old_correct = old_accept & (p.argmax(1)==labels) & (labels>0)
    new_correct = new_accept & (output.argmax(1)==labels) & (labels>0)

    def identities(mask):
        return [dict(index=int(i), source=rows[i]['source_image'], sha256=rows[i]['sha256'], view=rows[i]['view'],
            label=rows[i]['label'], before=p[i].tolist(), after=output[i].tolist()) for i in np.flatnonzero(mask)]

    eligible = improves(before, after) and improves(unique_before, unique_after)
    # Retain all-fit parameters only for reproducibility; not a deployment export.
    np.savez_compressed(RUN/'final.npz', **fit(x, labels, p))
    np.savez_compressed(RUN/'predictions.npz', before=p, after=output, labels=labels, folds=folds)
    result = dict(before=before, after=after, duplicate_aware=dict(before=unique_before, after=unique_after),
        lost_true_views=identities(old_correct & ~new_correct), gained_true_views=identities(~old_correct & new_correct),
        unique_lost_true_views=int(np.sum((old_correct & ~new_correct)[unique])),
        unique_gained_true_views=int(np.sum((~old_correct & new_correct)[unique])),
        rows=len(rows), unique_crop_views=len(unique), source_groups=len(set(groups)), folds=receipts,
        eligible_for_development=eligible, protocol_sha256=sha256(PROTOCOL), training_sha256=sha256(TRAINING),
        script_sha256=sha256(__file__), helper_sha256=sha256('scripts/blue_soft_shape_fusion.py'),
        input_verifier_sha256=sha256('scripts/audit_balloon_hog_gate.py'),
        predictions=dict(path=str(RUN/'predictions.npz'), sha256=sha256(RUN/'predictions.npz')),
        final_model=dict(path=str(RUN/'final.npz'), sha256=sha256(RUN/'final.npz')),
        scope=protocol['scope'], development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False)
    write_json(OUTPUT, result)
    print({k: result[k] for k in ('before', 'after', 'duplicate_aware', 'unique_lost_true_views',
        'unique_gained_true_views', 'eligible_for_development')}, flush=True)


if __name__ == '__main__':
    run()
