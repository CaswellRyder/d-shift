"""Frozen-model research screen. Development mode evaluates only fixed anchored95."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from analyze_isolated_replay import matched_truth
from research_orange_supplement import digest, supplement
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import proposals, suppress_duplicates


def counts(observations, targets, label):
    predictions = [r['box'] for r in observations if r['accepted'] and r['label'] == label]
    truth = [r['box'] for r in targets
             if '_'.join(r['label'].lower().split()[:2]) == label]
    hits = len(matched_truth(predictions, truth))
    return dict(tp=hits, fp=len(predictions)-hits, fn=len(truth)-hits)


def admit_supplements(baseline, supplemental, threshold):
    """Preserve baseline decisions; extras may neither replace nor overlap them.

    Threshold is an experimental score cutoff, not a calibrated probability.
    Cross-class overlap and containment reject conflicting supplemental boxes.
    """
    if not 0 <= threshold <= 1:
        raise ValueError('Invalid supplemental threshold')
    accepted = [dict(r) for r in baseline if r['accepted']]
    for row in sorted(supplemental, key=lambda r: -r['score']):
        if not row['accepted'] or not row['label'].startswith('orange_') or row['score'] < threshold:
            continue
        def conflicts(old):
            a, b = row['box'], old['box']
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
                0, min(a[3], b[3])-max(a[1], b[1]))
            smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
            return iou(a, b) > .3 or (smaller > 0 and intersection/smaller > .6)
        if not any(conflicts(old) for old in accepted):
            accepted.append(dict(row))
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development-manifest', type=Path,
                        help='Explicit development_validation manifest; no policy sweep')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    if audit['split'] != 'train' or audit['test_evaluated']:
        raise ValueError('Train-only audit required')
    root = Path('data/roboflow-dtr-v10-grouped/coco/train')
    if digest(root/'_annotations.coco.json') != audit['annotation_sha256']:
        raise ValueError('Annotations changed')
    if digest('src/dtr/vision.py') != audit['vision_sha256']:
        raise ValueError('Proposal code changed')
    predictor = Predictor(args.model, allow_unvalidated=True)
    classes = predictor.metadata['classes'][1:]
    cutoffs = (.95,) if args.development_manifest else (.8, .9, .95)
    methods = ('baseline', 'anchored_95') if args.development_manifest else (
        'baseline', 'trial', 'anchored_80', 'anchored_90', 'anchored_95')
    frames = audit['frames']
    if args.development_manifest:
        manifest = json.loads(args.development_manifest.read_text())
        if manifest['split'] != 'development_validation' or manifest['size'] != [320, 240]:
            raise ValueError('Require 320x240 development_validation inputs')
        root = args.development_manifest.parent
        frames = [dict(file=r['path'], sha256=r['sha256'], targets=[
            dict(label=' '.join(t['label'].split('_')).title()+' Goal', box=t['box'])
            for t in r['truth'] if t['label'] in classes]) for r in manifest['frames']]
    totals = {method: {label: dict(tp=0, fp=0, fn=0) for label in classes}
              for method in methods}
    rows = []
    source = audit['supplemental_source']
    for index, frame in enumerate(frames):
        path = root/frame['file']
        if digest(path) != frame['sha256']:
            raise ValueError('Input changed')
        with Image.open(path) as opened:
            rgb = cv2.resize(np.asarray(opened.convert('RGB')), (320, 240))
        baseline = proposals(rgb, 'goal', limit=12, profile='balloon_components')
        local = proposals(rgb, 'goal', limit=12 if source == 'orange_local' else 64,
                          profile=source)
        trial = supplement(baseline, local)
        predictions = []
        for candidate in trial:
            x1, y1, x2, y2 = candidate['crop_box']
            predictions.append({**candidate, **predictor.predict(rgb[y1:y2, x1:x2])})
        row = dict(file=frame['file'], baseline_count=len(baseline), raw_predictions=predictions)
        base_obs = suppress_duplicates(predictions[:len(baseline)], policy='nested')
        policies = dict(baseline=base_obs)
        if not args.development_manifest:
            policies['trial'] = suppress_duplicates(predictions, policy='nested')
        for cutoff in cutoffs:
            policies[f'anchored_{round(cutoff*100)}'] = admit_supplements(
                base_obs, predictions[len(baseline):], cutoff)
        for method, observations in policies.items():
            row[method] = [r for r in observations if r['accepted']]
            for label in classes:
                for key, value in counts(observations, frame['targets'], label).items():
                    totals[method][label][key] += value
        rows.append(row)
        if index % 100 == 0:
            print(index, '/', len(frames), flush=True)
    for labels in totals.values():
        for value in labels.values():
            value['precision'] = value['tp']/max(1, value['tp']+value['fp'])
            value['recall'] = value['tp']/max(1, value['tp']+value['fn'])
    report = dict(scope=__doc__, model_sha256=digest(args.model),
                  metadata_sha256=digest(args.model.with_suffix('.json')),
                  audit_sha256=digest(args.audit), verifier_sha256=digest(__file__),
                  split='development_validation' if args.development_manifest else 'train',
                  development_manifest_sha256=digest(args.development_manifest)
                  if args.development_manifest else None,
                  test_evaluated=False, deployment_approved=False,
                  threshold=predictor.metadata['threshold'],
                  supplemental_thresholds=list(cutoffs),
                  inference_policy='stateless 12 vs up to 14 crops; NOT four-crop temporal',
                  totals=totals, frames=rows)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
