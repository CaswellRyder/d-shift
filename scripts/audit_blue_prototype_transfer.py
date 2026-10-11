"""Leave-one-background-source-out TRAIN diagnostic; no radius retuning."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_prototype_rejection import adjust, correction, fit
from scripts.fit_blue_prototype_rejection import OUTPUT as TRAINING, checked_inputs

OUTPUT = Path('research/balloon-red-blue-20261008/blue-prototype-transfer-20261011.json')


def fold_bank(features, labels, sources, centers, logits, center_sources, excluded):
    sources, center_sources = np.asarray(sources), np.asarray(center_sources)
    keep, keep_centers = sources != excluded, center_sources != excluded
    if (keep.shape != (len(features),) or keep_centers.shape != (len(centers),)
            or keep.all() or keep_centers.all() or not keep_centers.any()):
        raise ValueError('Require a removed source and remaining prototypes')
    return fit(features[keep], labels[keep], centers[keep_centers], logits[keep_centers]), keep, keep_centers


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = read_json(TRAINING)
    rows, labels, _, extra, _, _ = checked_inputs()
    if (report['training']['sha256'] != sha256(report['training']['path'])
            or report['script_sha256'] != sha256('scripts/fit_blue_prototype_rejection.py')
            or any(sha256(p) != h for p,h in report['helpers_sha256'].items())):
        raise ValueError('Training evidence changed')
    with np.load(report['training']['path']) as archive:
        data = dict(archive)
    if not np.array_equal(labels, data['labels']):
        raise ValueError('TRAIN labels changed')
    sources = [r['source_image'] for r in rows]
    center_sources = [r['source_image'] for r in extra]
    folds, decisions = [], []
    for group in sorted(set(center_sources)):
        bank, keep, keep_centers = fold_bank(data['features'], labels, sources,
            data['center_features'], data['center_logits'], center_sources, group)
        withheld = np.flatnonzero(~keep_centers)
        delta = correction(data['center_features'][withheld], bank)
        before = softmax(data['center_logits'][withheld])
        after = adjust(before, delta)
        positive_delta = correction(data['features'][keep][labels[keep] > 0], bank)
        if np.any(positive_delta != 0):
            raise ValueError('Fold no longer protects TRAIN positives')
        folds.append(dict(source=group, removed_curriculum_rows=int((~keep).sum()),
            removed_prototypes=len(withheld), remaining_prototypes=int(keep_centers.sum()),
            remaining_positive_rows=int((labels[keep]>0).sum()),
            scale_fitted_without_withheld_source=True, prototypes_fitted_without_withheld_source=True))
        for index, old, new, value in zip(withheld,before,after,delta,strict=True):
            decisions.append(dict(prototype_index=int(index),source=group,sha256=extra[index]['sha256'],
                correction=float(value),before=old.tolist(),after=new.tolist(),
                background_before=bool(old.argmax()==0),background_after=bool(new.argmax()==0)))
    if sorted(d['prototype_index'] for d in decisions) != list(range(12)):
        raise ValueError('Incomplete source-held-out diagnostic')
    write_json(OUTPUT,dict(folds=folds,decisions=decisions,sources=len(folds),crops=len(decisions),
        corrected_crops=sum(d['correction']>0 for d in decisions),
        background_before=sum(d['background_before'] for d in decisions),
        background_after=sum(d['background_after'] for d in decisions),
        training_sha256=sha256(TRAINING),script_sha256=sha256(__file__),
        helper_sha256=sha256('scripts/blue_prototype_rejection.py'),
        scope='Retrospective TRAIN-source holdout diagnostic. Refit only fixed prototype/scaling recipe after removing every row and prototype from each held-out source. Frozen backbone is shared. No hyperparameter sweep, IMG fitting or qualification claim.',
        test_evaluated=False,deployment_approved=False,qualification_achieved=False))
    print(dict(sources=len(folds),corrected=sum(d['correction']>0 for d in decisions),
        backgrounds_passed=sum(d['background_after'] for d in decisions)),flush=True)


if __name__ == '__main__':
    audit()
