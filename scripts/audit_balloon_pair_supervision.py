"""Audit TRAIN-only evidence for learned containment; never fit or relabel data.

Counts are source coverage, not accuracy. Pair preferences are automatically
derived diagnostic labels, not an approved new training dataset.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from dtr.tracking import iou
from dtr.vision import suppress_duplicates
from scripts.audit_public_balloon_export import bounded_read
from scripts.compare_balloon_containment import area, nested
from scripts.research_balloon_search import experimental
from scripts.train_balloon_quality import training_rows

BOOTSTRAP_SHA = '6c080dfe38a876b73c9f9e342e3158ec0ce1e830fd77b4a2056663b2641d9191'
LABELS = ('red_balloon','blue_balloon')


def training_photos(manifest, archive_path):
    doc = read_json(manifest)
    if (sha256(manifest) != BOOTSTRAP_SHA or sha256(archive_path) != doc['source_archive_sha256']
            or doc.get('training_approved') is not True or doc.get('synthetic') is not False):
        raise ValueError('Original reviewed training source changed')
    held = [r for r in doc['samples'] if r['split'] != 'train']
    groups = defaultdict(dict)
    for row in doc['samples']:
        if row['split'] != 'train' or row['label'] not in LABELS:
            continue
        source = row['source_image']
        if (not source.startswith('balloon/train/') or '..' in Path(source).parts
                or Path(source).name.lower().startswith('img_')
                or any(row[k] == h.get(k) for h in held for k in ('source_image','source_sha256'))):
            raise ValueError('Forbidden or held-out training source')
        previous = groups[source].setdefault(row['annotation_id'],row)
        if any(previous[k] != row[k] for k in ('label','box_xyxy','source_sha256')):
            raise ValueError('Conflicting annotation views')
    frames = []
    with zipfile.ZipFile(archive_path) as archive:
        for source, annotations in sorted(groups.items()):
            with Image.open(io.BytesIO(bounded_read(archive,source,20_000_000))) as opened:
                im = opened.convert('RGB')
                digest = hashlib.sha256(im.tobytes()).hexdigest()
                if any(r['source_sha256'] != digest for r in annotations.values()):
                    raise ValueError('Original source pixels changed')
                rgb = cv2.resize(np.asarray(im),(320,240),interpolation=cv2.INTER_AREA)
                truth = [dict(label=r['label'],box=[v*(320/im.width if k%2==0 else 240/im.height)
                    for k,v in enumerate(r['box_xyxy'])],id=r['annotation_id']) for r in annotations.values()]
            frames.append(dict(source=source,source_group=digest,domain='matterport',rgb=rgb,
                               proposals=experimental(rgb,'mser'),truth=truth))
    return frames


def reviewed_indoor(config):
    rows, receipts = training_rows(config)
    frames = []
    for entry in config['sources']:
        root = Path(entry['queue']).resolve()
        queue = read_json(root/'review.json')
        for frame in queue['frames']:
            path = (root/frame['path']).resolve()
            if not path.is_relative_to(root) or sha256(path) != frame['sha256']:
                raise ValueError('Indoor frame path or pixels changed')
            candidates = [r for r in rows if r['source_image'] == 'train/'+frame['source']]
            if not candidates:
                continue
            with Image.open(path) as im:
                rgb = np.asarray(im.convert('RGB'))
            if rgb.shape != (240,320,3):
                raise ValueError('Unexpected reviewed scan size')
            frames.append(dict(source='train/'+frame['source'],source_group=candidates[0]['source_group'],
                domain='engdes2',rgb=rgb,proposals=candidates,
                truth=[dict(t,id=i) for i,t in enumerate(frame['quality_targets'])]))
    return frames,receipts


def preference(parent, child, truth):
    """Both boxes must overlap the same unambiguous reviewed target.

    A nonmatching proposal is not automatically a background or a balloon part.
    """
    matches = []
    for row in (parent,child):
        scores = sorted([(iou(row['box'],t['box']),i) for i,t in enumerate(truth)
                         if t['label'] == row['label']],reverse=True)
        if not scores or scores[0][0] <= 0:
            return None,'unmatched'
        if len(scores)>1 and abs(scores[0][0]-scores[1][0]) < 1e-8:
            return None,'ambiguous_target'
        matches.append(scores[0])
    if matches[0][1] != matches[1][1]:
        return None,'different_targets'
    a,b = matches[0][0],matches[1][0]
    if not min(a,b) < .5 <= max(a,b) or abs(a-b) < .1:
        return None,'no_clear_preference'
    return dict(choice='parent' if a>b else 'child',parent_iou=a,child_iou=b,
                target_id=truth[matches[0][1]]['id']),'admitted_for_audit'


def frame_pairs(frame, predictor):
    found = []
    for proposal in frame['proposals']:
        a,b,c,d = proposal['crop_box']
        found.append(dict(proposal,**predictor.predict(frame['rgb'][b:d,a:c])))
    found = suppress_duplicates(found)
    pairs,excluded = [],Counter()
    for i,a in enumerate(found):
        for j,b in enumerate(found[i+1:],i+1):
            if not (a['accepted'] and b['accepted'] and a['label'] == b['label'] and nested(a['box'],b['box'])):
                continue
            (pi,parent),(ci,child) = sorted(((i,a),(j,b)),key=lambda p:area(p[1]['box']),reverse=True)
            label,reason = preference(parent,child,frame['truth'])
            if label is None:
                excluded[reason] += 1
                continue
            pairs.append(dict(label,source=frame['source'],source_group=frame['source_group'],domain=frame['domain'],
                label=parent['label'],parent_index=pi,child_index=ci,parent_box=parent['box'],child_box=child['box'],
                parent_score=parent['score'],child_score=child['score']))
    return pairs,dict(excluded)


def coverage(pairs):
    result = {}
    for choice in ('parent','child'):
        rows = [r for r in pairs if r['choice'] == choice]
        groups = Counter(r['source_group'] for r in rows)
        result[choice] = dict(pairs=len(rows),source_groups=len(groups),
            largest_source_fraction=max(groups.values())/len(rows) if rows else None,
            by_domain=dict(Counter(r['domain'] for r in rows)),by_color=dict(Counter(r['label'] for r in rows)),
            groups=dict(sorted(groups.items())))
    return result


def audit(manifest, archive, quality_config, models):
    frames = training_photos(manifest,archive)
    indoor,receipts = reviewed_indoor(read_json(quality_config))
    frames += indoor
    results = {}
    for name,path in models.items():
        predictor = Predictor(path,allow_unvalidated=True)
        m = predictor.metadata
        if m['classes'] != ['background',*LABELS] or m['size'] != 64 or m['threshold'] != .8 or m['task'] != 'balloon':
            raise ValueError('Require fixed 64-pixel red/blue student at threshold 0.8')
        pairs,exclusions = [],Counter()
        for frame in frames:
            found,excluded = frame_pairs(frame,predictor)
            pairs.extend(found)
            exclusions.update(excluded)
        counts = coverage(pairs)
        # Planning floor only: passing this does not certify labels or accuracy.
        sufficient = all(counts[k]['source_groups'] >= 10 for k in ('parent','child'))
        results[name] = dict(model_sha256=sha256(path),metadata_sha256=sha256(Path(path).with_suffix('.json')),
            coverage=counts,excluded=dict(exclusions),pairs=pairs,
            source_coverage_floor_met=sufficient,minimum_sources_per_choice=10,
            pair_labels_reviewed=False,training_approved=False)
    return dict(results=results,frames=dict(Counter(f['domain'] for f in frames)),
        manifest_sha256=sha256(manifest),archive_sha256=sha256(archive),
        quality_config_sha256=sha256(quality_config),quality_reviews=receipts,
        script_sha256=sha256(__file__),helper_sha256={p:sha256(p) for p in
            ('scripts/train_balloon_quality.py','scripts/research_balloon_search.py',
             'scripts/compare_balloon_containment.py','src/dtr/vision.py','src/dtr/runtime.py')},
        scope='Training-only supervision coverage after frozen classifier and ordinary NMS; not accuracy',
        rules='Same-class accepted nested pair, same known target, IoU straddles 0.5 with margin >=0.1',
        caveat='Incomplete labels; source groups are not independently verified recording sessions',
        training_approved=False,test_evaluated=False,deployment_approved=False,pi_timing_measured=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = audit('data/balloon-red-blue-bootstrap-20261008/manifest.json',
        'data/raw/matterport-balloon/balloon_dataset.zip','configs/balloon-quality-training-20261008.json',
        {f'seed{seed}':f'runs/balloon-red-blue-new-views-fixed-teacher{suffix}-20261008/student.fp32.tflite'
         for seed,suffix in ((42,''),(43,'-seed43'))})
    write_json(args.output,report)
    print({k:dict(coverage=v['coverage'],excluded=v['excluded']) for k,v in report['results'].items()})
