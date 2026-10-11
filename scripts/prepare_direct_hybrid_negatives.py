"""Queue conservative TRAIN-only crop false alarms for explicit visual review."""
from pathlib import Path

from PIL import Image, ImageDraw

from dtr.data import read_json, sha256, write_json
from scripts.audit_scale_balanced_search import checked_panels
from scripts.audit_direct_hybrid_balloon import checked_inputs, OUTPUT as AUDIT

AUDIT_SHA = '9c382a7ac43b376d0bf13240a21734e7fbb414204ac1dfb24d5efeddacdc6689'
ROOT = Path('data/direct-hybrid-negative-review-20261011')


def clear_of_targets(box, truth, guard=16):
    if guard != 16 or len(box) != 4 or not 0 <= box[0] < box[2] <= 320 or not 0 <= box[1] < box[3] <= 240:
        raise ValueError('Require bounded crop and fixed16pixel uncertainty guard')
    a, b, c, d = box
    return not any(min(c, t['box'][2]+guard) > max(a, t['box'][0]-guard)
                   and min(d, t['box'][3]+guard) > max(b, t['box'][1]-guard) for t in truth)


def prepare():
    if ROOT.exists():
        raise FileExistsError('Preserve review queue')
    checked_inputs()
    if sha256(AUDIT) != AUDIT_SHA:
        raise ValueError('Source audit changed')
    audit = read_json(AUDIT)
    if sha256(audit['raw']['path']) != audit['raw']['sha256']:
        raise ValueError('Source predictions changed')
    raw, panels = read_json(audit['raw']['path']), checked_panels()
    chosen = {}
    for panel, sources in panels.items():
        for index, (frame, rgb) in enumerate(sources):
            if 'IMG_' in frame['source'] or not frame['source'].startswith(('train/', 'openimages/train/')):
                raise ValueError('Only admitted TRAIN scenes allowed')
            for seed in (47, 48):
                predicted = raw[panel][f'base_admit{seed}']['frames'][index]
                if any(predicted[k] != frame[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                    raise ValueError('Scene identity mismatch')
                for proposal_index, row in enumerate(predicted['detections']):
                    if (not row['raw_accepted'] or row['label'] == 'background'
                            or not clear_of_targets(row['crop_box'], frame['truth'])):
                        continue
                    # Bound correlated negatives: one highest-confidence crop per scene/color.
                    key = frame['source'], row['label']
                    record = dict(panel=panel, source=frame['source'], frame_sha256=frame['frame_sha256'],
                        truth=frame['truth'], seed=seed, proposal_index=proposal_index,
                        predicted_label=row['label'], score=row['score'], scores=row['scores'],
                        box=row['box'], crop_box=row['crop_box'])
                    if key not in chosen or row['score'] > chosen[key][0]['score']:
                        chosen[key] = record, rgb
    if not chosen:
        raise ValueError('No conservative candidates; do not invent negatives')
    ROOT.mkdir(parents=True)
    samples = []
    tile_w, tile_h, columns = 384, 212, 3
    sheet = Image.new('RGB', (columns*tile_w, ((len(chosen)+columns-1)//columns)*tile_h), 'white')
    for index, (record, rgb) in enumerate(chosen.values()):
        frame = Image.fromarray(rgb)
        crop = frame.crop(record['crop_box'])
        frame_path, crop_path = f'frame-{index:02}.png', f'crop-{index:02}.png'
        frame.save(ROOT/frame_path)
        crop.save(ROOT/crop_path)
        display = frame.copy()
        ImageDraw.Draw(display).rectangle(record['crop_box'], outline='lime', width=2)
        tile = Image.new('RGB', (tile_w, tile_h), 'white')
        tile.paste(display.resize((256, 192)), (0, 20))
        tile.paste(crop.resize((128, 128)), (256, 40))
        ImageDraw.Draw(tile).text((3, 3), f"{index}: {record['panel']} {record['predicted_label']} {record['score']:.3f}", fill='black')
        sheet.paste(tile, ((index % columns)*tile_w, (index//columns)*tile_h))
        samples.append(dict(record, id=index, split='train', frame_path=frame_path,
            rendered_frame_sha256=sha256(ROOT/frame_path), crop_path=crop_path,
            crop_sha256=sha256(ROOT/crop_path), proposed_label='background',
            visual_review='pending', training_approved=False))
    sheet.save(ROOT/'sheet.png')
    write_json(ROOT/'review.json', dict(samples=samples, audit_sha256=AUDIT_SHA,
        raw_sha256=audit['raw']['sha256'], script_sha256=sha256(__file__),
        sheet_sha256=sha256(ROOT/'sheet.png'), guard_pixels=16,
        scope='Complete reviewed TRAIN scenes only; crops outside all annotated red/blue targets plus16pixel guard. Geometric screening is not visual approval. One candidate per scene/predicted color, no IMG_ or held-out mining.',
        training_approved=False, development_evaluated=False, test_evaluated=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(samples=len(samples), queue=str(ROOT/'review.json'), sheet=str(ROOT/'sheet.png')), flush=True)


if __name__ == '__main__':
    prepare()
