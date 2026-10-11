"""Known TRAIN targets only; no development sweep or independent precision claim."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json,sha256,write_json
from dtr.runtime import Predictor
from scripts.audit_balloon_scene_errors import attribute
from scripts.balloon_dense_runtime import DenseProposals
from scripts.train_balloon_dense_proposals import checked_data
from scripts.research_balloon_search import experimental
from scripts.research_predictor import classify_candidates
from scripts.pi_balloon_search_bench import select_detections
from scripts.research_contained_refinement import refine
from scripts.evaluate_balloon_dense_scenes import CLASSIFIER

PROTOCOL='configs/balloon-contained-refinement-20261011.json'
BASE='data/balloon-dense-complete-training-20261011'
TRIAL='research/balloon-red-blue-20261008/dense-public-trial-20261011.json'


def matches(truth,detections):
    targets=attribute(dict(truth=truth,detections=detections))
    return {i for i,r in enumerate(targets) if r['reason']=='detected'}


def probe(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    doc,_,_,folds,_=checked_data(BASE)
    receipt=read_json(TRIAL)
    networks={}
    for mode in ('control','admit'):
        for seed in ('47','48'):
            path=Path(receipt['trials'][mode][seed]['path'])/'proposal.fp32.tflite'
            if sha256(path)!=receipt['trials'][mode][seed]['model_sha256']:
                raise ValueError('Probe localizer changed')
            networks[mode+seed]=DenseProposals(path)
    classifier=Predictor(CLASSIFIER,allow_unvalidated=True)
    results={name:[] for name in networks}
    for row,fold in zip(doc['frames'],folds,strict=True):
        if fold==0:
            continue
        with Image.open(Path(BASE)/row['path']) as im:
            rgb=np.asarray(im.convert('RGB'))
        base=select_detections(classify_candidates(classifier,rgb,experimental(rgb,'mser')),
                               'split_merged',2.42913421076193)
        before=matches(row['truth'],base)
        for name,network in networks.items():
            called=any(r['accepted'] and r['label']=='blue_balloon' for r in base)
            learned=network.predict(rgb) if called else []
            after=refine(base,learned)
            if len(base)!=len(after):
                raise ValueError('Refinement added detections')
            for a,b in zip(base,after,strict=True):
                if any(a[k]!=b[k] for k in ('label','score','scores','accepted','crop_box')):
                    raise ValueError('Refinement changed classification evidence')
                if a['label']!='blue_balloon' and a!=b:
                    raise ValueError('Refinement changed non-blue detection')
            detected=matches(row['truth'],after)
            results[name].append(dict(source=row['source'],frame_sha256=row['sha256'],truth=row['truth'],
                dense_called=called,baseline_detections=base,refined_detections=after,
                baseline_target_ids=sorted(before),refined_target_ids=sorted(detected),
                gained=sorted(detected-before),lost=sorted(before-detected),
                replaced_boxes=sum('pre_refinement_box' in r for r in after)))
    totals={}
    for name,frames in results.items():
        counts={}
        for label in ('red_balloon','blue_balloon'):
            counts[label]=dict(targets=sum(t['label']==label for f in frames for t in f['truth']),
                baseline_tp=sum(f['truth'][i]['label']==label for f in frames for i in f['baseline_target_ids']),
                refined_tp=sum(f['truth'][i]['label']==label for f in frames for i in f['refined_target_ids']),
                gained=sum(f['truth'][i]['label']==label for f in frames for i in f['gained']),
                lost=sum(f['truth'][i]['label']==label for f in frames for i in f['lost']))
        totals[name]=dict(counts=counts,frames=len(frames),
            dense_calls=sum(f['dense_called'] for f in frames),
            changed_frames=sum(f['replaced_boxes']>0 for f in frames),
            replaced_boxes=sum(f['replaced_boxes'] for f in frames))
    feasible=all(totals['admit'+s]['counts']['blue_balloon']['gained']>0
                 and totals['admit'+s]['counts']['blue_balloon']['lost']==0 for s in ('47','48'))
    write_json(output,dict(results=results,totals=totals,both_admit_seeds_training_gate_pass=feasible,
        base_manifest_sha256=sha256(Path(BASE)/'manifest.json'),trial_sha256=sha256(TRIAL),
        protocol_sha256=sha256(PROTOCOL),classifier_sha256=sha256(CLASSIFIER),
        script_sha256=sha256(__file__),helper_sha256=sha256('scripts/research_contained_refinement.py'),
        scope='Shared original fit frames only. Known-target recall diagnostic, not scene precision. Grouped holdout not inferred.',
        development_evaluated=False,test_evaluated=False,pi_tested=False,deployment_approved=False))
    print(totals)
    print(dict(both_admit_seeds_training_gate_pass=feasible))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    probe(p.parse_args().output)
