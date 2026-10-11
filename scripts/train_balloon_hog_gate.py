"""Fixed five-fold source-grouped TRAIN shape gate; no development access."""
import hashlib
from pathlib import Path
import time

import numpy as np

from dtr.data import read_json,sha256,write_json
from dtr.runtime import Predictor,softmax
from scripts.balloon_hog_gate import features,fold_fit,source_folds,fit,score
from scripts.compare_current_blue_search import MODEL,MODEL_SHA
from scripts.fit_blue_prototype_rejection import checked_inputs

PROTOCOL = Path('configs/balloon-hog-gate-20261011.json')
RUN = Path('runs/balloon-hog-gate-20261011')
OUTPUT = Path('research/balloon-red-blue-20261008/balloon-hog-gate-training-20261011.json')


def counts(labels, probabilities, accepted):
    predicted = probabilities.argmax(axis=1)
    result = {}
    for i,name in ((1,'red_balloon'),(2,'blue_balloon')):
        tp = int(np.sum(accepted & (predicted==i) & (labels==i)))
        fp = int(np.sum(accepted & (predicted==i) & (labels!=i)))
        fn = int(np.sum((labels==i) & ~(accepted & (predicted==i))))
        p, r = tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.
        result[name] = dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)
    return result


def run():
    if RUN.exists() or OUTPUT.exists():
        raise FileExistsError(RUN)
    p = read_json(PROTOCOL)
    if (p['ridge_penalty'] != .1 or p['scale_floor'] != .001 or p['folds'] != 5 or p['training_rows'] != 2074
            or sha256(MODEL) != MODEL_SHA):
        raise ValueError('Fixed recipe/model changed')
    rows,labels,pixels,extra,center_pixels,previous = checked_inputs()
    groups,folds = source_folds(rows)
    if set(folds) != set(range(5)):
        raise ValueError('Incomplete source folds')
    start = time.perf_counter()
    x = features(pixels)
    feature_seconds = time.perf_counter()-start
    classifier = Predictor(MODEL,allow_unvalidated=True)
    probabilities = []
    for image in pixels:
        classifier.interpreter.set_tensor(classifier.input['index'],image[None])
        classifier.interpreter.invoke()
        probabilities.append(softmax(classifier.interpreter.get_tensor(classifier.output['index'])[0]))
    probabilities = np.asarray(probabilities)
    baseline_accept = (probabilities.argmax(axis=1)>0)&(probabilities.max(axis=1)>=.8)
    oof,thresholds,receipts = np.empty(len(rows)),np.empty(len(rows)),[]
    RUN.mkdir(parents=True)
    for fold in range(5):
        model, values, keep = fold_fit(x,labels,folds,fold)
        if set(np.asarray(groups)[keep]) & set(np.asarray(groups)[~keep]):
            raise ValueError('Source group leaks across fold')
        if np.any(score(x[keep],model)[labels[keep]>0] < model['threshold']-1e-12):
            raise ValueError('Training-only threshold changed fitted-positive retention')
        oof[~keep],thresholds[~keep] = values,float(model['threshold'])
        path = RUN/f'fold-{fold}.npz'
        np.savez_compressed(path,**model)
        receipts.append(dict(fold=fold,fit_rows=int(keep.sum()),held_out_rows=int((~keep).sum()),
            fit_groups=len(set(np.asarray(groups)[keep])),held_out_groups=len(set(np.asarray(groups)[~keep])),
            threshold=float(model['threshold']),model_sha256=sha256(path)))
    veto = baseline_accept & (probabilities.argmax(axis=1)==2) & (oof<thresholds)
    before = counts(labels,probabilities,baseline_accept)
    after = counts(labels,probabilities,baseline_accept & ~veto)
    correct = baseline_accept & (probabilities.argmax(axis=1)==labels) & (labels>0)
    lost = np.flatnonzero(correct & veto)
    neg = []
    for r,image in zip(extra,center_pixels,strict=True):
        matched = [i for i,old in enumerate(rows) if old['sha256']==r['sha256'] and old['view']=='original']
        if not matched or not np.array_equal(pixels[matched[0]],image) or len(set(folds[matched])) != 1:
            raise ValueError('Reviewed clutter source does not share fold')
        i = matched[0]
        neg.append(dict(source=r['source_image'],sha256=r['sha256'],fold=int(folds[i]),
            parent_accepted_blue=bool(baseline_accept[i] and probabilities[i].argmax()==2),
            veto=bool(veto[i]),shape_score=float(oof[i]),threshold=float(thresholds[i])))
    minimum_ok = all(min(after[c]['precision'],after[c]['recall']) >= min(before[c]['precision'],before[c]['recall']) for c in before)
    progress = (after['blue_balloon']['fp'] < before['blue_balloon']['fp']
        and after['blue_balloon']['f1'] > before['blue_balloon']['f1'] and minimum_ok)
    # Final all-TRAIN model retained for reproducibility, not automatically deployed.
    final = fit(x,labels)
    np.savez_compressed(RUN/'final.npz',**final)
    np.savez_compressed(RUN/'predictions.npz',features=x,labels=labels,probabilities=probabilities,
        folds=folds,oof=oof,thresholds=thresholds,veto=veto)
    write_json(OUTPUT,dict(before=before,after=after,folds=receipts,rows=len(rows),source_groups=len(set(groups)),
        unique_crop_hashes=len({r['sha256'] for r in rows}), source_group_ids=groups,
        lost_true_crop_views=[dict(index=int(i),source=rows[i]['source_image'],sha256=rows[i]['sha256'],view=rows[i]['view'],
            label=rows[i]['label'],fold=int(folds[i]),shape_score=float(oof[i]),threshold=float(thresholds[i])) for i in lost],
        reviewed_backgrounds=neg, vetoed_crop_views=int(veto.sum()),
        eligible_for_development=bool(progress),min_per_color_metric_nonregressing=minimum_ok,
        training_source=dict(path=str(previous),sha256=sha256(previous)),
        feature_tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        model_sha256=MODEL_SHA,metadata_sha256=sha256(MODEL.with_suffix('.json')),
        predictions=dict(path=str(RUN/'predictions.npz'),sha256=sha256(RUN/'predictions.npz')),
        final_model=dict(path=str(RUN/'final.npz'),sha256=sha256(RUN/'final.npz')),
        protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        helper_sha256=sha256('scripts/balloon_hog_gate.py'),
        desktop_feature_seconds=feature_seconds,
        scope='Source-grouped out-of-fold TRAIN crop feasibility; correlated source sessions and repeated crop views remain. Original neural classifier was fitted on prior TRAIN data. Not detector accuracy, independently held-out qualification or Pi timing.',
        test_evaluated=False,development_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(before=before,after=after,lost=len(lost),reviewed_backgrounds_vetoed=sum(r['veto'] for r in neg),
        eligible_for_development=bool(progress)),flush=True)


if __name__ == '__main__':
    run()
