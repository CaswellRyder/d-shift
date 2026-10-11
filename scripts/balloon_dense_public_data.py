"""Licensed public localization supervision with unknown regions masked out."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from scripts.openimages_balloon_training import checked_extra, QUEUE, REVIEW, LICENSE
from scripts.prepare_balloon_dense_proposals import encode

PROTOCOL = 'configs/balloon-dense-public-trial-20261011.json'


def encode_public(truth, backgrounds):
    """Retain only explicit negative regions compatible with known positive cells."""
    accepted,skipped = [],[]
    for box in backgrounds:
        try:
            encode(truth,accepted+[box])
        except ValueError as error:
            if str(error) != 'Explicit background overlaps supervised balloon cell':
                raise
            skipped.append(box)
        else:
            accepted.append(box)
    target,collisions = encode(truth,accepted)
    return target,accepted,skipped,collisions


def public_frames():
    approved = checked_extra()
    queue = read_json(Path(QUEUE)/'review.json')
    grouped = defaultdict(list)
    for r in approved:
        grouped[r['frame_id']].append(r)
    records = []
    for index,rows in sorted(grouped.items()):
        frame = queue['frames'][index]
        truth,backgrounds = [],[]
        for r in rows:
            if r['label']=='background':
                backgrounds.append(r['box'] if r['kind']=='source_balloon_box' else r['crop_box'])
            else:
                if r['kind'] != 'source_balloon_box':
                    raise ValueError('Positive localization needs a reviewed source box')
                truth.append(dict(box=r['box'],label=r['label']))
        target,negative,skipped,collisions = encode_public(truth,backgrounds)
        if not target[:,:,2:4].any():
            continue
        with Image.open(Path(QUEUE)/frame['path']) as im:
            rgb = np.asarray(im.convert('RGB'))
        records.append(dict(source='openimages/train/'+frame['image_id'],
            source_sha256=frame['original_sha256'],panel='openimages-reviewed-training',
            source_path=str(Path(QUEUE)/frame['path']),sha256=frame['sha256'],
            truth=truth,negative_regions=negative,skipped_negative_regions=skipped,
            reviewed_crop_ids=[r['id'] for r in rows],source_frame_id=index,
            supervised_heat_cells=int((target[:,:,2:4]>0).sum()),box_cells=int(target[:,:,-1].sum()),
            box_collisions=collisions,rgb=rgb,target=target))
    if not records:
        raise ValueError('No usable public localization supervision')
    return records


def expected(mode):
    if mode not in ('control','admit'):
        raise ValueError('Require control or admit')
    from scripts.train_balloon_dense_proposals import checked_data
    protocol = read_json(PROTOCOL)
    root = Path(protocol['base_data'])
    if sha256(root/'manifest.json') != protocol['base_manifest_sha256']:
        raise ValueError('Dense baseline data changed')
    base,x,y,folds,groups = checked_data(root)
    public = public_frames()
    # Keep the original grouped holdout closed to every added control sample.
    choices = np.random.default_rng(47).choice(np.flatnonzero(folds!=0),len(public),replace=False)
    images,targets,records = list(x),list(y),[]
    sources = []
    output_folds,output_groups = folds.tolist(),list(groups)
    for i,row in enumerate(base['frames']):
        records.append(dict(row,path=f'base/{i:03d}.png'))
        sources.append(root/row['path'])
    for i,(row,chosen) in enumerate(zip(public,choices,strict=True)):
        if mode=='admit':
            record = {k:v for k,v in row.items() if k not in ('rgb','target','source_path')}
            record.update(path=f'extra/{i:03d}.png',expansion_kind='public')
            images.append(cv2.resize(row['rgb'],(160,120),interpolation=cv2.INTER_AREA).astype(np.float32))
            targets.append(row['target'])
            sources.append(Path(row['source_path']))
            output_folds.append(1)
            output_groups.append(row['source'])
        else:
            original = base['frames'][int(chosen)]
            record = dict(original,path=f'extra/{i:03d}.png',expansion_kind='resampled_fit',base_index=int(chosen))
            images.append(x[chosen])
            targets.append(y[chosen])
            sources.append(root/original['path'])
            output_folds.append(int(folds[chosen]))
            output_groups.append(groups[chosen])
        records.append(record)
    return dict(base=base,records=records,images=np.asarray(images,np.float32),targets=np.asarray(targets),
        folds=np.asarray(output_folds),groups=output_groups,sources=sources,public=public,
        public_expansion=dict(mode=mode,protocol_sha256=sha256(PROTOCOL),
            base_manifest_sha256=protocol['base_manifest_sha256'],base_frames=len(base['frames']),
            extra_frames=len(public),review_sha256=sha256(REVIEW),queue_sha256=sha256(Path(QUEUE)/'review.json'),
            license_verification_sha256=sha256(LICENSE),control_indices=choices.tolist(),
            data_helper_sha256=sha256(__file__)))


def checked_public_data(root):
    root = Path(root)
    doc = read_json(root/'manifest.json')
    if (doc.get('training_approved') is not True or doc.get('deployment_approved') is not False
            or doc.get('test_evaluated') is not False or doc['grid'] != [15,20]):
        raise ValueError('Require training-only public expansion')
    source = expected(doc['public_expansion']['mode'])
    if doc['public_expansion'] != source['public_expansion'] or doc['frames'] != source['records']:
        raise ValueError('Expansion supervision or source identity changed')
    if sha256(root/'targets.npy') != doc['targets_sha256']:
        raise ValueError('Public targets changed')
    saved = np.load(root/'targets.npy',allow_pickle=False)
    if not np.array_equal(saved,source['targets']):
        raise ValueError('Targets differ from explicit reviewed geometry')
    for row in doc['frames']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != row['sha256']:
            raise ValueError('Changed expansion pixels or unsafe path')
    fit,hold = source['folds']!=0,source['folds']==0
    for key in ('source','source_sha256'):
        if {r[key] for r,f in zip(doc['frames'],fit) if f} & {r[key] for r,f in zip(doc['frames'],hold) if f}:
            raise ValueError('Public expansion crosses grouped holdout')
    return doc,source['images'],saved,source['folds'],source['groups']


def prepare(mode,output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = expected(mode)
    output.mkdir(parents=True)
    (output/'base').mkdir()
    (output/'extra').mkdir()
    for row,path in zip(source['records'],source['sources'],strict=True):
        shutil.copyfile(path,output/row['path'])
    np.save(output/'targets.npy',source['targets'],allow_pickle=False)
    doc = dict(frames=source['records'],targets_sha256=sha256(output/'targets.npy'),grid=[15,20],
        public_expansion=source['public_expansion'],script_sha256=sha256(__file__),
        scope='Public TRAIN expansion with sparse known regions; original grouped holdout unchanged. Not independent qualification.',
        training_approved=True,test_evaluated=False,deployment_approved=False)
    write_json(output/'manifest.json',doc)
    checked_public_data(output)
    base_n=source['public_expansion']['base_frames']
    return dict(mode=mode,frames=len(doc['frames']),extra_frames=len(doc['frames'])-base_n,
        extra_boxes=dict(Counter(t['label'] for f in doc['frames'][base_n:] for t in f['truth'])),
        extra_heat_cells=int(source['targets'][base_n:,:,:,2:4].sum()),
        extra_box_collisions=sum(f['box_collisions'] for f in doc['frames'][base_n:]),
        manifest_sha256=sha256(output/'manifest.json'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('control','admit'),required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(prepare(a.mode,a.output))
