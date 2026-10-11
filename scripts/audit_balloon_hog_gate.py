"""Reproduce TRAIN HOG evidence, including duplicate-aware error counts."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor, softmax
from scripts.balloon_hog_gate import features, fit, fold_fit, score, source_folds
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.train_balloon_hog_gate import OUTPUT as TRAINING, PROTOCOL, RUN, counts

OUTPUT = Path('research/balloon-red-blue-20261008/balloon-hog-gate-audit-20261011.json')


def unique_indices(rows, labels, pixels):
    """Deduplicate exact crop/view pairs; conflicting identities fail closed."""
    seen, result = {}, []
    for i, row in enumerate(rows):
        key = (row['sha256'], row['view'])
        if key in seen:
            old = seen[key]
            if labels[old] != labels[i] or not np.array_equal(pixels[old], pixels[i]):
                raise ValueError('Repeated crop/view has conflicting pixels or labels')
        else:
            seen[key] = i
            result.append(i)
    return np.array(result, dtype=int)


def verified_data():
    report = read_json(TRAINING)
    required = {
        str(PROTOCOL): report['protocol_sha256'],
        'scripts/train_balloon_hog_gate.py': report['script_sha256'],
        'scripts/balloon_hog_gate.py': report['helper_sha256'],
        str(MODEL): MODEL_SHA,
        str(MODEL.with_suffix('.json')): report['metadata_sha256'],
    }
    for key in ('training_source', 'predictions', 'final_model'):
        required[report[key]['path']] = report[key]['sha256']
    required.update({str(RUN/f"fold-{r['fold']}.npz"): r['model_sha256'] for r in report['folds']})
    if report['model_sha256'] != MODEL_SHA or any(sha256(p) != h for p, h in required.items()):
        raise ValueError('HOG experiment provenance changed')
    rows, labels, pixels, _, _, _ = checked_inputs()
    with np.load(report['predictions']['path']) as archive:
        saved = dict(archive)
    x = features(pixels)
    groups, folds = source_folds(rows)
    if (groups != report['source_group_ids'] or len(rows) != report['rows']
            or not np.array_equal(labels, saved['labels']) or not np.array_equal(folds, saved['folds'])
            or not np.array_equal(x, saved['features'])
            or hashlib.sha256(x.tobytes()).hexdigest() != report['feature_tensor_sha256']):
        raise ValueError('TRAIN inputs/features/folds changed')
    return report, rows, labels, pixels, x, groups, folds, saved


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report, rows, labels, pixels, x, groups, folds, saved = verified_data()
    oof, thresholds = np.empty(len(rows)), np.empty(len(rows))
    errors = []
    for fold in range(5):
        model, values, keep = fold_fit(x, labels, folds, fold)
        if set(np.asarray(groups)[keep]) & set(np.asarray(groups)[~keep]):
            raise ValueError('Source overlap')
        with np.load(RUN/f'fold-{fold}.npz') as old:
            error = max(float(np.max(np.abs(model[k]-old[k]))) for k in model)
        if error > 1e-10:
            raise ValueError('Fold refit changed')
        errors.append(error)
        oof[~keep], thresholds[~keep] = values, model['threshold']
    final = fit(x, labels)
    with np.load(RUN/'final.npz') as old:
        final_error = max(float(np.max(np.abs(final[k]-old[k]))) for k in final)
    if final_error > 1e-10 or not np.allclose(oof, saved['oof'], atol=1e-10, rtol=0):
        raise ValueError('Final model or out-of-fold scores changed')
    threshold_error = float(np.max(np.abs(thresholds-saved['thresholds'])))
    if threshold_error > 1e-10:
        raise ValueError('Thresholds changed')
    classifier = Predictor(MODEL, allow_unvalidated=True)
    probabilities = []
    for image in pixels:
        classifier.interpreter.set_tensor(classifier.input['index'], image[None])
        classifier.interpreter.invoke()
        probabilities.append(softmax(classifier.interpreter.get_tensor(classifier.output['index'])[0]))
    p = np.asarray(probabilities)
    probability_error = float(np.max(np.abs(p-saved['probabilities'])))
    if probability_error > 1e-7:
        raise ValueError('Frozen classifier outputs changed')
    accepted = (p.argmax(axis=1)>0) & (p.max(axis=1)>=.8)
    veto = accepted & (p.argmax(axis=1)==2) & (oof<thresholds)
    correct = accepted & (p.argmax(axis=1)==labels) & (labels>0)
    lost = np.flatnonzero(correct & veto)
    if (not np.array_equal(veto, saved['veto'])
            or list(lost) != [r['index'] for r in report['lost_true_crop_views']]
            or counts(labels, p, accepted) != report['before']
            or counts(labels, p, accepted & ~veto) != report['after']):
        raise ValueError('Reported decisions/counts changed')
    unique = unique_indices(rows, labels, pixels)
    before, after = counts(labels[unique], p[unique], accepted[unique]), counts(labels[unique], p[unique], (accepted & ~veto)[unique])
    result = dict(training_sha256=sha256(TRAINING), script_sha256=sha256(__file__),
        rows=len(rows), unique_crop_views=len(unique), source_groups=len(set(groups)),
        refit_max_errors=errors, final_refit_max_error=final_error, classifier_max_error=probability_error,
        threshold_max_error=threshold_error,
        fitted_positive_retention=bool(np.all(score(x, final)[labels>0] >= final['threshold'])),
        duplicate_aware=dict(before=before, after=after, lost_true_views=int(np.sum((correct & veto)[unique])),
            removed_false_views=int(np.sum((~correct & veto)[unique]))),
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False,
        decision='Reject hard shape gate: source-held-out TRAIN recall/F1 regress; no development or deployment advancement.',
        scope='Audit of TRAIN source-grouped shape-head feasibility. Parent neural classifier was trained on TRAIN; not end-to-end held-out accuracy.')
    write_json(OUTPUT, result)
    print(result, flush=True)


if __name__ == '__main__':
    audit()
