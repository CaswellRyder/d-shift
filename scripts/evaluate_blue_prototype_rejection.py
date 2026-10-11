"""Frozen-proposal prototype feasibility; TRAIN reproduction before development."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor, softmax
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.blue_prototype_rejection import adjust, correction, fit
from scripts.build_blue_peak_blend import checked_scenes
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_blue_peak_blend import OUTPUT as BASELINE, RAW as BASELINE_RAW
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.evaluate_red_blue_development import LABELS, detection_counts, metrics
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.fit_blue_prototype_rejection import PROTOCOL, OUTPUT as TRAINING, checked_inputs, feature_model
from scripts.pi_current_balloon_pipeline import compare
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.replay_balloon_suppression import raw
from scripts.research_balloon_confidence_selection import confidence_selection
from scripts.summarize_balloon_mask_convergence import meets_target
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/blue-prototype-scenes-20261011.json')
RAW = Path('runs/blue-prototype-scenes-20261011.json')


def checked_bank(model):
    receipt = read_json(TRAINING)
    if (receipt['protocol_sha256'] != sha256(PROTOCOL)
            or receipt['script_sha256'] != sha256('scripts/fit_blue_prototype_rejection.py')
            or receipt['training_gate_pass'] is not True
            or any(sha256(p) != h for p,h in receipt['helpers_sha256'].items())
            or any(sha256(receipt[k]['path']) != receipt[k]['sha256'] for k in ('bank','training','previous_provenance'))):
        raise ValueError('Prototype training changed or failed')
    _, labels, pixels, _, centers, _ = checked_inputs()
    with np.load(receipt['training']['path']) as archive:
        arrays = dict(archive)
    with np.load(receipt['bank']['path']) as archive:
        bank = dict(archive)
    rebuilt = fit(arrays['features'], labels, arrays['center_features'], arrays['center_logits'])
    if (not np.array_equal(labels, arrays['labels'])
            or any(not np.array_equal(bank[k],v) for k,v in rebuilt.items())):
        raise ValueError('Prototype fit does not reproduce')
    delta = 0.
    for x, fields in ((pixels, ('features','logits')), (centers, ('center_features','center_logits'))):
        for actual, key in zip(model(x, training=False), fields, strict=True):
            delta = max(delta,float(np.max(np.abs(actual.numpy()-arrays[key]))))
    if delta > 1e-5:
        raise ValueError('TRAIN pixel-to-feature mapping changed')
    recalculated = correction(arrays['features'], bank)
    center_delta = correction(arrays['center_features'], bank)
    if (not np.array_equal(recalculated, arrays['delta']) or np.any(recalculated[labels>0] != 0)
            or not np.array_equal(center_delta, arrays['center_delta'])
            or not np.array_equal(adjust(softmax(arrays['logits']),recalculated), arrays['after'])
            or not np.array_equal(adjust(softmax(arrays['center_logits']),center_delta), arrays['center_after'])
            or np.any(arrays['center_after'].argmax(axis=1) != 0)):
        raise ValueError('Training correction/protection changed')
    return bank, delta


def change_predictions(found, features, bank):
    delta = correction(features, bank)
    scores = adjust([r['scores'] for r in found], delta)
    result = []
    names = ('background','red_balloon','blue_balloon')
    for row, values, amount in zip(found, scores, delta, strict=True):
        if amount == 0:
            result.append(dict(row))
            continue
        index = int(values.argmax())
        result.append(dict(row, label=names[index], score=float(values[index]), scores=values.tolist(),
            accepted=bool(index != 0 and values[index] >= .8)))
    return result, delta


def evaluate():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError(OUTPUT)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    feature_net = feature_model()
    bank, train_delta = checked_bank(feature_net)
    doc, saved = checked_scenes()
    if sha256(MODEL) != MODEL_SHA or doc['classifier_sha256'] != MODEL_SHA:
        raise ValueError('Base classifier changed')
    predictor = Predictor(MODEL, allow_unvalidated=True)
    train = {s['source']:(s['source_file_sha256'],s['truth'],rgb)
        for s,rgb in source_frames() if s['origin'] == 'complete_fit'}
    if len(train) != 64:
        raise ValueError('TRAIN scene coverage changed')
    panels, result = {}, {}
    max_score_delta, max_feature_model_score_delta = 0., 0.
    for panel in ['training']+[p for p in saved if p != 'training']:
        base = saved[panel]['background/blend']
        if checked_metrics(base) != doc['panels'][panel]['background/blend']['metrics']:
            raise ValueError('Baseline scene counts changed')
        pixels = train
        if panel != 'training':
            frames, review = reviewed_frames('data/'+panel,'configs/'+panel+'.json')
            if review['threshold'] != .8 or review['matching_iou'] != .5:
                raise ValueError('Development operating point changed')
            pixels = {s['source']:(s['sha256'],s['provisional_truth'],rgb) for s,rgb in frames}
        rows, corrected = [], 0
        for old in base['frames']:
            digest, truth, rgb = pixels[old['source']]
            if digest != old['frame_sha256'] or truth != old['truth']:
                raise ValueError('Scene identity changed')
            found = raw(old['detections'])
            if confidence_selection(found) != old['detections']:
                raise ValueError('Unchanged selection did not reproduce')
            crops, measured = [], []
            for row in found:
                a,b,c,d = row['crop_box']
                crop = rgb[b:d,a:c]
                prediction = predictor.predict(crop)
                measured.append(dict(row, **prediction))
                crops.append(np.asarray(Image.fromarray(crop).resize((64,64),Image.Resampling.BILINEAR),np.float32))
            score_delta, _ = compare(measured, found)
            max_score_delta = max(max_score_delta,score_delta)
            delta = np.zeros(len(found))
            if crops:
                features, logits = [v.numpy() for v in feature_net(np.stack(crops),training=False)]
                max_feature_model_score_delta = max(max_feature_model_score_delta,
                    float(np.max(np.abs(softmax(logits)-np.asarray([r['scores'] for r in found])))))
                candidate, delta = change_predictions(found,features,bank)
            else:
                candidate = []
            if max_feature_model_score_delta > 1e-5:
                raise ValueError('Feature parent does not reproduce classifier')
            selected = confidence_selection(candidate)
            corrected += int((delta > 0).sum())
            rows.append(dict({k:old[k] for k in ('id','source','frame_sha256','truth')},
                detections=selected, counts=detection_counts(truth,selected), prototype_corrections=delta.tolist(),
                classified_proposals=len(found)))
        totals = {c:{k:sum(r['counts'][c][k] for r in rows) for k in ('tp','fp','fn')} for c in LABELS}
        result[panel] = dict(frames=rows, metrics=metrics(totals))
        checked_metrics(result[panel])
        panels[panel] = dict(metrics=result[panel]['metrics'], **compare_frames(rows,base['frames']),
            nonregression=nonregression(base['metrics'],result[panel]['metrics']),
            corrected_proposals=corrected, meets_94=meets_target(result[panel]['metrics']))
        print(panel,dict(metrics=panels[panel]['metrics'],corrected=corrected,
            lost=len(panels[panel]['lost_parent_targets'])),flush=True)
    improved = any(panels[p]['metrics'][c]['fp'] < saved[p]['background/blend']['metrics'][c]['fp']
        for p in panels if p != 'training' for c in LABELS)
    retained = all(p['nonregression'] and not p['lost_parent_targets'] for p in panels.values())
    write_json(RAW,result)
    write_json(OUTPUT,dict(panels=panels, raw=dict(path=str(RAW),sha256=sha256(RAW)),
        baseline_sha256=sha256(BASELINE), baseline_raw_sha256=sha256(BASELINE_RAW), training_sha256=sha256(TRAINING),
        script_sha256=sha256(__file__), protocol_sha256=sha256(PROTOCOL),
        helper_sha256=sha256('scripts/blue_prototype_rejection.py'), max_parent_score_delta=max_score_delta,
        max_feature_model_score_delta=max_feature_model_score_delta, max_train_feature_delta=train_delta,
        development_fp_improved=improved, no_target_or_metric_regression=retained,
        eligible_for_single_pass_export=improved and retained,
        scope='Fixed-proposal offboard feasibility only. Existing TFLite classifier plus separate frozen Keras feature extraction is not a Pi implementation or a speed result. Repeated related development, not independent qualification.',
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_single_pass_export=improved and retained),flush=True)


if __name__ == '__main__':
    evaluate()
