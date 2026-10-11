"""Recount all scenes and inspect TRAIN-only peak switching behind target losses."""
from pathlib import Path

import cv2
import numpy as np

from dtr.data import read_json,sha256,write_json
from dtr.tracking import iou
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.balloon_dense_runtime import DenseProposals
from scripts.balloon_grid_runtime import GridProposals
from scripts.evaluate_blue_head_only import OUTPUT as SCENES,RAW,BASE
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.summarize_public_balloon_trial import checked_metrics


def peak_cells(values):
    if values.shape != (15,20,6) or not np.isfinite(values).all():
        raise ValueError('Require finite localizer grid')
    heat = 1/(1+np.exp(-np.clip(values[...,1],-80,80)))
    mask = (heat >= .1)&(heat == cv2.dilate(heat,np.ones((3,3),np.uint8)))
    rows = []
    for y,x in np.argwhere(mask):
        ox,oy,bw,bh = values[y,x,2:]
        cx,cy,w,h = (x+ox)*16,(y+oy)*16,bw*320,bh*240
        box = [max(0.,cx-w/2),max(0.,cy-h/2),min(320.,cx+w/2),min(240.,cy+h/2)]
        if min(box[2]-box[0],box[3]-box[1]) >= 4:
            rows.append(dict(cell=[int(x),int(y)],score=float(heat[y,x]),box=list(map(float,box))))
    return rows


def audit():
    output = BASE/'blue-head-only-decision-20261011.json'
    if output.exists():
        raise FileExistsError(output)
    doc = read_json(SCENES)
    if (doc['raw']['sha256'] != sha256(RAW) or doc['script_sha256'] != sha256('scripts/evaluate_blue_head_only.py')
            or any(sha256(p) != h for p,h in doc['helpers_sha256'].items())
            or any(sha256(p) != h for p,h in doc['inputs'].items())):
        raise ValueError('Evaluation sources changed')
    full = read_json(RAW)
    for panel,variants in full.items():
        for key,result in variants.items():
            saved = doc['panels'][panel][key]
            compared = compare_frames(result['frames'],variants['parent']['frames'])
            if (checked_metrics(result) != saved['metrics'] or any(saved[k] != v for k,v in compared.items())
                    or saved['nonregression'] != nonregression(variants['parent']['metrics'],result['metrics'])):
                raise ValueError('Counts or identity-level losses changed')
    networks = {}
    for name,cls in (('parent',GridProposals),('background',DenseProposals)):
        identity = doc['models'][name]
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Localizer changed')
        networks[name] = cls(path)
    losses = doc['panels']['training']['background']['lost_parent_targets']
    source_names = {r['source'] for r in losses}
    pixels = {s['source']:(s,rgb) for s,rgb in source_frames()
        if s['origin'] == 'complete_fit' and s['source'] in source_names}
    diagnostics = []
    for loss in losses:
        source,rgb = pixels[loss['source']]
        truth = loss['target']
        grids,frames,peaks = {},{},{}
        for name,model in networks.items():
            frame = next(f for f in full['training'][name]['frames'] if f['source'] == loss['source'])
            if frame['frame_sha256'] != source['source_file_sha256'] or frame['truth'] != source['truth']:
                raise ValueError('TRAIN diagnostic pixels changed')
            frames[name] = frame
            model.predict(rgb)
            grids[name] = model.interpreter.get_tensor(model.output['index'])[0]
            peaks[name] = peak_cells(grids[name])
        geometry_delta = float(np.max(np.abs(grids['parent'][...,2:]-grids['background'][...,2:])))
        red_delta = float(np.max(np.abs(grids['parent'][...,0]-grids['background'][...,0])))
        if geometry_delta > 1e-5 or red_delta > 1e-5:
            raise ValueError('Fixed-grid invariance failed on lost scene')
        evidence = {}
        for name,frame in frames.items():
            rows = []
            for d in frame['detections']:
                overlap = iou(truth['box'],d['box'])
                if overlap < .5:
                    continue
                matches = [r for r in peaks[name] if np.max(np.abs(np.asarray(r['box'])-d['box'])) < .001]
                rows.append(dict(box=d['box'],crop_box=d['crop_box'],label=d['label'],score=d['score'],
                    accepted=d['accepted'],raw_accepted=d['raw_accepted'],target_iou=overlap,blue_peak=matches))
            evidence[name] = rows
        diagnostics.append(dict(source=loss['source'],target=truth,frame_sha256=source['source_file_sha256'],
            geometry_max_delta=geometry_delta,red_heat_max_delta=red_delta,overlapping_detections=evidence,
            parent_blue_peaks=peaks['parent'],candidate_blue_peaks=peaks['background']))
    eligible = all(p['background']['nonregression'] and not p['background']['lost_parent_targets'] for p in doc['panels'].values())
    if eligible != doc['eligible_for_pi_comparison']:
        raise ValueError('Candidate gate changed')
    write_json(output,dict(panels=doc['panels'],training_loss_diagnostics=diagnostics,
        inputs={str(SCENES):sha256(SCENES),str(RAW):sha256(RAW)},script_sha256=sha256(__file__),
        eligible_for_pi_comparison=eligible,
        decision='Eligible only for isolated research timing' if eligible else 'Reject head-only candidate: loses a TRAIN-scene target; keep incumbent.',
        scope='All scene counts recounted. Additional diagnostic uses only already-approved TRAIN pixels. Per-cell geometry invariance does not guarantee identical selected peak cells or crop boxes.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(eligible=eligible,diagnostics=diagnostics))


if __name__ == '__main__':
    audit()
