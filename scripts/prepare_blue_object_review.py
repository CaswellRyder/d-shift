"""Review queue from licensed novel TRAIN objects; predictions never label it."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.acquire_blue_objects import OUTPUT as ROOT
from scripts.prepare_blue_source_review import references
from scripts.prepare_openimages_balloon_review import extra_references, checked_path, sheets
from scripts.prepare_indoor_balloon_review import fingerprint, holdout_distance
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.balloon_grid_runtime import GridProposals
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_balloon_search import experimental

OUTPUT = Path('data/blue-object-review-20261011')
SEARCH = Path('runs/balloon-grid-convergence-coarse-seed47-20261011/proposal.fp32.tflite')


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    acquisition,licenses = read_json(ROOT/'acquisition.json'),read_json(ROOT/'license-verification.json')
    if licenses['acquisition_sha256']!=sha256(ROOT/'acquisition.json') or sha256(MODEL)!=MODEL_SHA:
        raise ValueError('Source/classifier changed')
    if acquisition['script_sha256']!=sha256('scripts/acquire_blue_objects.py'):
        raise ValueError('Acquisition changed')
    for key,name in (('selection_sha256','selection.json'),('attribution_sha256','attribution.json')):
        if acquisition[key]!=sha256(ROOT/name):
            raise ValueError('Acquisition metadata changed')
    if any(sha256(p)!=h for p,h in acquisition['helpers_sha256'].items()):
        raise ValueError('Acquisition helpers changed')
    verified = {r['image_id'] for r in licenses['results'] if r['verified'] is True}
    prints,receipts = references()
    extra_references(prints,receipts)
    for p,h in read_json(ROOT/'selection.json')['excluded_acquisitions'].items():
        if sha256(p)!=h:
            raise ValueError('Prior source changed')
        receipts[p]=h
        for r in read_json(p)['images']:
            with Image.open(checked_path(Path(p).parent,r['path'],r['sha256'])) as image:
                prints.append(fingerprint(np.asarray(image.convert('RGB'))))
    predictor,search = Predictor(MODEL,allow_unvalidated=True),GridProposals(SEARCH)
    frames,samples,rejected,selected,counters = [],[],[],[],Counter()
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'frames').mkdir()
    (OUTPUT/'crops').mkdir()
    for source in acquisition['images']:
        if source['image_id'] not in verified:
            rejected.append(dict(image_id=source['image_id'],reason='License not verified'))
            continue
        if source['attribution']['Subset']!='train':
            raise ValueError('Non-TRAIN source')
        with Image.open(checked_path(ROOT,source['path'],source['sha256'])) as image:
            rgb = np.asarray(image.convert('RGB'))
        distance = holdout_distance(rgb,prints)
        internal = holdout_distance(rgb,selected) if selected else None
        if distance<=8 or (internal is not None and internal<=8):
            rejected.append(dict(image_id=source['image_id'],reason='Perceptual duplicate',
                reference_hamming=distance,selected_hamming=internal))
            continue
        scan = cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA)
        candidates = supplement(experimental(scan,'mser'),search.predict(scan))
        pool=[]
        for c in candidates:
            a,b,d,e = c['crop_box']
            pred = predictor.predict(scan[b:e,a:d])
            counters['classified']+=1
            if pred['label']!='background' and pred['score']>=.5:
                pool.append(dict(proposal=c,mining_prediction=pred))
        chosen = sorted(pool,key=lambda r:-r['mining_prediction']['score'])[:2]
        counters['no_review_candidates']+=int(not chosen)
        if not chosen:
            continue
        i=len(frames)
        path=f'frames/{i:03d}.png'
        Image.fromarray(scan).save(OUTPUT/path)
        frames.append(dict(id=i,path=path,sha256=sha256(OUTPUT/path),image_id=source['image_id'],
            original_path=source['path'],original_sha256=source['sha256'],category=source['category'],
            boxes=[],source_annotations=source['annotations'],attribution=source['attribution'],
            reference_hamming=distance,selected_hamming=internal))
        selected.append(fingerprint(rgb))
        for row in chosen:
            a,b,c,d=row['proposal']['crop_box']
            path=f'crops/{len(samples):03d}.png'
            Image.fromarray(scan[b:d,a:c]).save(OUTPUT/path)
            samples.append(dict(row,id=len(samples),frame_id=i,path=path,sha256=sha256(OUTPUT/path),
                source_image='openimages/train/'+source['image_id'],split='train',label='unreviewed'))
    sheets(OUTPUT,'frames',frames,3,(320,270))
    sheets(OUTPUT,'crops',samples,6,(160,160))
    def identity(path):
        return dict(path=str(path),sha256=sha256(path),metadata_sha256=sha256(path.with_suffix('.json')))
    write_json(OUTPUT/'review.json',dict(frames=frames,samples=samples,rejected=rejected,counters=dict(counters),
        acquisition_sha256=sha256(ROOT/'acquisition.json'),license_sha256=sha256(ROOT/'license-verification.json'),
        script_sha256=sha256(__file__),reference_inputs=receipts,
        helpers_sha256={p:sha256(p) for p in ('scripts/prepare_blue_source_review.py',
            'scripts/prepare_openimages_balloon_review.py','scripts/research_balloon_search.py',
            'scripts/audit_balloon_dense_proposals.py','scripts/balloon_grid_runtime.py')},
        classifier=identity(MODEL),search=identity(SEARCH),
        sheets={p.name:sha256(p) for p in OUTPUT.glob('*.jpg')},
        training_approved=False,review_required=True,heldout_pixels_used_for_duplicate_screening_only=True,
        test_evaluated=False,deployment_approved=False,
        scope='Bounded current-runtime crops from independent public TRAIN sources. Source classes and high model scores are not labels. Visual crop and frame review required.'))
    print(dict(frames=len(frames),crops=len(samples),rejected=len(rejected),counts=dict(counters)))


if __name__=='__main__':
    prepare()
