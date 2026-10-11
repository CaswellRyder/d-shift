"""Quarantine current red/blue errors from new-source licensed TRAIN imagery."""
from collections import Counter
from pathlib import Path

from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import Predictor
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.prepare_blue_false_alarm_review import source_frames
from scripts.prepare_grid_clutter_review import guarded_background
from scripts.prepare_openimages_balloon_review import sheets
from scripts.balloon_grid_runtime import GridProposals
from scripts.research_balloon_search import experimental
from scripts.research_censored_ellipse import proposals as censored

OUTPUT = Path('data/current-color-clutter-review-20261011')
PREVIOUS = ('data/blue-false-alarm-review-20261011/review.json',
            'data/blue-clutter-round2-review-20261011/review.json')


def select(pool):
    per_source, per_color, chosen = Counter(), Counter(), []
    for r in sorted(pool,key=lambda r:(-r['mining_prediction']['score'],r['source'],r['crop_box'])):
        color = r['mining_prediction']['label']
        if per_source[r['source']] >= 2 or per_color[color] >= 24:
            continue
        per_source[r['source']] += 1
        per_color[color] += 1
        chosen.append(r)
    return chosen


def prepare():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('Mining classifier changed')
    excluded = {r['source'] for p in PREVIOUS for r in read_json(p)['frames']}
    old = read_json(PREVIOUS[0])
    identities = old['models']
    models = {}
    for name,identity in identities.items():
        path = Path(identity['path'])
        if sha256(path) != identity['sha256'] or sha256(path.with_suffix('.json')) != identity['metadata_sha256']:
            raise ValueError('Localizer changed')
        models[name] = GridProposals(path)
    predictor = Predictor(MODEL,allow_unvalidated=True)
    sources, pixels, pool, counters = {}, {}, {}, Counter()
    for source,rgb in source_frames():
        key = source['source']
        if key in excluded:
            counters['previous_source_excluded'] += 1
            continue
        sources[key],pixels[key] = source,rgb
        counters[source['origin']+'_frames'] += 1
        candidates = [(mode,c) for mode in ('mser','mser_blue_ellipse') for c in experimental(rgb,mode)]
        candidates += [('censored',c) for c in censored(rgb)]
        candidates += [(name,c) for name,model in models.items() for c in model.predict(rgb)]
        seen = set()
        for name,c in candidates:
            box = c['crop_box']
            counters['proposals'] += 1
            if not guarded_background(box,[{'box':b} for b in source['boxes']]):
                counters['guarded_out'] += 1
                continue
            crop_key = key,tuple(box)
            if crop_key in seen:
                if crop_key in pool:
                    pool[crop_key]['searches'].append(name)
                continue
            seen.add(crop_key)
            a,b,d,e = box
            prediction = predictor.predict(rgb[b:e,a:d])
            counters['classified'] += 1
            if prediction['label'] == 'background' or prediction['score'] < .8:
                continue
            pool[crop_key] = dict(source=key,crop_box=box,proposal_box=c['box'],searches=[name],
                mining_prediction={k:prediction[k] for k in ('label','score','scores','accepted')})
    chosen = select(pool.values())
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'frames').mkdir()
    (OUTPUT/'crops').mkdir()
    frames,samples,frame_ids = [],[],{}
    for row in chosen:
        key = row['source']
        if key not in frame_ids:
            i = len(frames)
            frame_ids[key] = i
            path = f'frames/{i:03d}.png'
            Image.fromarray(pixels[key]).save(OUTPUT/path)
            frames.append(dict(sources[key],id=i,path=path,sha256=sha256(OUTPUT/path)))
        a,b,c,d = row['crop_box']
        path = f'crops/{len(samples):03d}.png'
        Image.fromarray(pixels[key][b:d,a:c]).save(OUTPUT/path)
        samples.append(dict(row,id=len(samples),frame_id=frame_ids[key],path=path,sha256=sha256(OUTPUT/path),
            split='train',label='unreviewed'))
    sheets(OUTPUT,'frames',frames,3,(320,270))
    sheets(OUTPUT,'crops',samples,6,(160,160))
    write_json(OUTPUT/'review.json',dict(frames=frames,samples=samples,models=identities,
        classifier=dict(path=str(MODEL),sha256=MODEL_SHA,metadata_sha256=sha256(MODEL.with_suffix('.json'))),
        previous_queues={p:sha256(p) for p in PREVIOUS},excluded_sources=sorted(excluded),
        diagnostics=dict(counters),pool_size=len(pool),guard_pixels=16,threshold=.8,
        script_sha256=sha256(__file__),helpers_sha256={p:sha256(p) for p in (
            'scripts/prepare_blue_false_alarm_review.py','scripts/prepare_grid_clutter_review.py',
            'scripts/research_balloon_search.py','scripts/research_censored_ellipse.py')},
        sheets={p.name:sha256(p) for p in sorted(OUTPUT.glob('*.jpg'))},
        training_approved=False,test_evaluated=False,deployment_approved=False,
        scope='TRAIN-only quarantine; all prior blue-review sources excluded including ambiguous samples. Prediction is not a label.'))
    print(dict(selected=len(samples),frames=len(frames),pool=len(pool),diagnostics=dict(counters),
        queue_sha256=sha256(OUTPUT/'review.json')))


if __name__ == '__main__':
    prepare()
