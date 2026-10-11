"""Freeze exact TRAIN false-alarm crops with original photographs for visual review."""
from pathlib import Path

from PIL import Image, ImageDraw

from dtr.data import read_json, sha256, write_json
from scripts.audit_scale_balanced_search import checked_panels
from scripts.prepare_direct_hybrid_negatives import clear_of_targets

AUDIT = Path('research/balloon-red-blue-20261008/positive-proposal-correction-20261011.json')
AUDIT_SHA = '428dcb4f78c2d84920c3e15670e76e6c7b0a5bf72139b880247707b4853d9f70'
ROOT = Path('data/source-resolved-negatives-20261011')
SELECTION = (
    ('window1', '22cd37f3c69ef8b8', 5, [85, 196, 112, 215]),
    ('window2', '3b83a4682fec0bc5', 1, [208, 220, 244, 238]),
    ('window2', '3dcd2b5a1e7fd9be', 11, [272, 88, 297, 99]),
)


def prepare():
    if ROOT.exists():
        raise FileExistsError('Preserve source-resolution review queue')
    if sha256(AUDIT) != AUDIT_SHA:
        raise ValueError('Mined audit changed')
    audit = read_json(AUDIT)
    if sha256(audit['raw']['path']) != audit['raw']['sha256']:
        raise ValueError('Mined predictions changed')
    saved, panels = read_json(audit['raw']['path']), checked_panels()
    samples = []
    ROOT.mkdir(parents=True)
    sheet = Image.new('RGB', (768, 260*len(SELECTION)), 'white')
    for index, (panel, image_id, proposal_index, crop_box) in enumerate(SELECTION):
        source = 'openimages/train/'+image_id
        frame, rgb = next((f, rgb) for f, rgb in panels[panel] if f['source'] == source)
        predicted = next(f for f in saved[panel]['corrected47_base_first']['frames'] if f['source'] == source)
        if any(frame[k] != predicted[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
            raise ValueError('Scene identity changed')
        proposal = predicted['detections'][proposal_index]
        if (proposal['crop_box'] != crop_box or proposal['proposal_origin'] != 'mser'
                or not proposal['accepted']):
            raise ValueError('Unexpected false-alarm crop')
        guard_clear = clear_of_targets(crop_box, frame['truth'])
        original = Path(f'data/raw/openimages-blue-{panel}-20261011/images/{image_id}.jpg')
        with Image.open(original) as im:
            im = im.convert('RGB')
            width, height = im.size
            source_box = [v*(width/320 if j % 2 == 0 else height/240) for j, v in enumerate(crop_box)]
            detail = im.crop(source_box)
        image = Image.fromarray(rgb)
        crop = image.crop(crop_box)
        crop_path, detail_path = ROOT/f'crop-{index}.png', ROOT/f'source-detail-{index}.png'
        crop.save(crop_path)
        detail.save(detail_path)
        ImageDraw.Draw(image).rectangle(crop_box, outline='lime', width=2)
        y = 260*index
        sheet.paste(image, (0, y+20))
        sheet.paste(crop.resize((192, 192)), (320, y+30))
        detail.thumbnail((256, 240))
        sheet.paste(detail, (512, y+20))
        ImageDraw.Draw(sheet).text((0, y), f'{index}: {image_id} / native inference crop / source detail', fill='black')
        samples.append(dict(id=index, panel=panel, source=source, split='train',
            frame_sha256=frame['frame_sha256'], truth=frame['truth'],
            proposal_index=proposal_index, crop_box=crop_box, box=proposal['box'],
            clear_of_16px_guard=guard_clear,
            crop_path=str(crop_path), crop_sha256=sha256(crop_path),
            original_path=str(original), original_sha256=sha256(original),
            original_dimensions=[width, height], original_crop_box=source_box,
            detail_path=str(detail_path), detail_sha256=sha256(detail_path),
            predicted_label=proposal['label'], score=proposal['score'],
            visual_review='pending', training_approved=False))
    sheet.save(ROOT/'sheet.png')
    write_json(ROOT/'review.json', dict(samples=samples, audit_sha256=AUDIT_SHA,
        raw_sha256=audit['raw']['sha256'], script_sha256=sha256(__file__),
        sheet_sha256=sha256(ROOT/'sheet.png'), training_approved=False,
        scope='Three exact false-alarm crops from admitted TRAIN photos. Original photographs resolve prior ambiguity; queue itself does not authorize labels. No held-out sources.'))
    print(dict(queue=str(ROOT/'review.json'), samples=len(samples)), flush=True)


if __name__ == '__main__':
    prepare()
