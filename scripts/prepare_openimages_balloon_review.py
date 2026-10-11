"""Screen public TRAIN candidates, retaining unreviewed color/object labels."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from scripts.prepare_blue_source_review import references, crop_box
from scripts.prepare_indoor_balloon_review import fingerprint, holdout_distance
from scripts.research_balloon_search import experimental

EXTRA_REFERENCES = (
    'engdes2-indoor-expansion3-review-20261011',
    'engdes2-indoor-expansion4-review-20261011',
    'astek-balloon-review-20261011', 'blue-source-review-20261011',
)


def checked_path(root, relative, digest):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or sha256(path) != digest:
        raise ValueError('Changed image or unsafe source path')
    return path


def extra_references(prints, receipts):
    for name in EXTRA_REFERENCES:
        root = Path('data')/name
        queue = root/'review.json'
        config = Path('configs')/(name+'.json')
        if read_json(config)['queue_sha256'] != sha256(queue):
            raise ValueError('Prior review changed')
        receipts[str(queue)],receipts[str(config)] = sha256(queue),sha256(config)
        for frame in read_json(queue)['frames']:
            path = checked_path(root,frame['path'],frame['sha256'])
            with Image.open(path) as im:
                prints.append(fingerprint(np.asarray(im.convert('RGB'))))


def sheets(output, kind, items, columns, tile):
    size = columns*4
    for start in range(0,len(items),size):
        sheet = Image.new('RGB',(columns*tile[0],4*tile[1]),'#ddd')
        draw = ImageDraw.Draw(sheet)
        for i,row in enumerate(items[start:start+size]):
            x,y = i%columns*tile[0],i//columns*tile[1]
            with Image.open(output/row['path']) as im:
                picture = im.convert('RGB')
            if kind == 'frames':
                mark = ImageDraw.Draw(picture)
                for box in row['boxes']:
                    mark.rectangle(box,outline='lime',width=1)
            sheet.paste(ImageOps.contain(picture,(tile[0]-4,tile[1]-25)),(x,y+25))
            title = f"{kind} {row['id']}"
            if kind == 'crops':
                title += f" / frame {row['frame_id']}"
            draw.text((x+2,y+2),title,fill='black')
        sheet.save(output/f'{kind}-{start:03d}.jpg')


def prepare(root, output):
    root,output = Path(root),Path(output)
    if output.exists():
        raise FileExistsError(output)
    acquisition = read_json(root/'acquisition.json')
    if acquisition['training_approved'] is not False or len(acquisition['images']) > 96:
        raise ValueError('Require bounded unapproved source')
    for key,path in (('selection_sha256','selection.json'),('attribution_sha256','attribution.json')):
        if acquisition[key] != sha256(root/path):
            raise ValueError('Source metadata changed')
    prints,receipts = references()
    extra_references(prints,receipts)
    print(f'{len(prints)} duplicate-screen references loaded',flush=True)
    output.mkdir(parents=True)
    for name in ('frames','crops'):
        (output/name).mkdir()
    frames,rows,rejected,selected = [],[],[],[]
    for source in acquisition['images']:
        path = checked_path(root,source['path'],source['sha256'])
        if source['attribution']['Subset'] != 'train':
            raise ValueError('Non-training source')
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
        distance = holdout_distance(rgb,prints)
        internal = holdout_distance(rgb,selected) if selected else None
        if distance <= 8 or (internal is not None and internal <= 8):
            rejected.append(dict(image_id=source['image_id'],reference_hamming=distance,selected_hamming=internal))
            continue
        scan = cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA)
        boxes,hypotheses = [],[]
        for n,annotation in enumerate(source['annotations']):
            box = [float(annotation[k])*scale for k,scale in
                   zip(('XMin','YMin','XMax','YMax'),(320,240,320,240))]
            boxes.append(box)
            if (annotation['IsGroupOf']=='0' and annotation['IsDepiction']=='0'
                    and min(box[2]-box[0],box[3]-box[1]) >= 4):
                hypotheses.append(dict(kind='source_balloon_box',box=box,
                    annotation_index=n,crop_box=crop_box(box)))
        hypotheses = hypotheses[:24]
        unmatched = [c for c in experimental(scan,'mser')
                     if max((iou(c['box'],b) for b in boxes),default=0) < .05]
        hypotheses += [dict(kind='unmatched_proposal',box=c['box'],crop_box=c['crop_box'],annotation_index=None)
                       for c in unmatched[:2]]
        index = len(frames)
        relative = f'frames/{index:03d}.png'
        Image.fromarray(scan).save(output/relative)
        frames.append(dict(id=index,image_id=source['image_id'],path=relative,sha256=sha256(output/relative),
            original_path=source['path'],original_sha256=source['sha256'],boxes=boxes,
            reference_hamming=distance,selected_hamming=internal,attribution=source['attribution']))
        for item in hypotheses:
            a,b,c,d = item['crop_box']
            if min(c-a,d-b) < 1:
                continue
            crop = f'crops/{len(rows):03d}.png'
            Image.fromarray(scan[b:d,a:c]).save(output/crop)
            rows.append(dict(item,id=len(rows),frame_id=index,path=crop,sha256=sha256(output/crop),
                split='train',label='unreviewed',source_image='openimages/train/'+source['image_id'],
                source_group='openimages:'+source['image_id']))
        selected.append(fingerprint(rgb))
    write_json(output/'review.json',dict(frames=frames,samples=rows,rejected=rejected,
        acquisition_path=str(root/'acquisition.json'),acquisition_sha256=sha256(root/'acquisition.json'),
        reference_inputs=receipts,script_sha256=sha256(__file__),training_approved=False,
        review_required=True,per_image_license_verified=False,label_mapping_approved=False,
        heldout_pixels_used_for_duplicate_screening_only=True,test_evaluated=False,
        caveat='Perceptual duplicate screen is heuristic. Source boxes may be incomplete; unmatched is not background approval.',
        deployment_approved=False))
    sheets(output,'frames',frames,3,(320,270))
    sheets(output,'crops',rows,6,(160,160))
    print(dict(frames=len(frames),crops=len(rows),rejected=len(rejected),queue_sha256=sha256(output/'review.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    prepare(a.root,a.output)
