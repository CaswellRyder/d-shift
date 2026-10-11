"""Diagnostic source-resolution overlays for manual scene annotation; no labels inferred."""
from pathlib import Path

from PIL import Image,ImageDraw

from dtr.data import read_json,sha256,write_json
from scripts.prepare_openimages_balloon_review import checked_path

QUEUE = Path('data/window2-complete-scene-review-20261011')
OUTPUT = Path('data/window2-annotation-details-20261011')


def render():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    q = read_json(QUEUE/'review.json')
    if sha256(QUEUE/'review.json')!='2de71bf9c8d7c2cbc672a8d4828a4d761ac3f6b0d3dedec3a8f304fc50245392':
        raise ValueError('Changed annotation queue')
    OUTPUT.mkdir(parents=True)
    regions = {6:[100,130,280,300],14:[0,0,680,370],25:[410,130,1024,390],29:[0,50,240,370]}
    result = []
    for fid,bounds in regions.items():
        f = next(r for r in q['frames'] if r['id']==fid)
        path = checked_path(Path('data/raw/openimages-blue-window2-20261011'),f['original_path'],f['original_sha256'])
        with Image.open(path) as im:
            raw = im.convert('RGB')
        marked = raw.copy()
        draw = ImageDraw.Draw(marked)
        for r in q['samples']:
            if r['frame_id'] != fid:
                continue
            a,b,c,d = [v*s for v,s in zip(r['box'],(raw.width/320,raw.height/240)*2)]
            draw.rectangle((a,b,c,d),outline='lime',width=2)
            draw.text((a+2,b+2),str(r['id']),fill='black',stroke_fill='white',stroke_width=2)
        raw.crop(bounds).resize(((bounds[2]-bounds[0])*2,(bounds[3]-bounds[1])*2)).save(OUTPUT/f'{fid}-plain.png')
        marked.crop(bounds).resize(((bounds[2]-bounds[0])*2,(bounds[3]-bounds[1])*2)).save(OUTPUT/f'{fid}-annotated.png')
        result.append(dict(frame_id=fid,original_sha256=f['original_sha256'],bounds_original_pixels=bounds))
    write_json(OUTPUT/'receipt.json',dict(queue_sha256=sha256(QUEUE/'review.json'),script_sha256=sha256(__file__),
        details=result,images_sha256={p.name:sha256(p) for p in OUTPUT.glob('*.png')},
        scope='Source-resolution inspection only; annotations remain unapproved. Coordinates in receipt refer to original photo, not320x240scan.'))


if __name__ == '__main__':
    render()
