"""Prediction-free completeness review of fit-only indoor balloon frames."""
import argparse
from pathlib import Path

from PIL import Image,ImageDraw

from dtr.data import sha256,write_json
from scripts.train_balloon_dense_proposals import checked_data


def prepare(root,output):
    root,output=Path(root),Path(output)
    if output.exists():
        raise FileExistsError(output)
    doc,_,_,folds,groups=checked_data(root)
    rows=[]
    for index,(frame,fold,group) in enumerate(zip(doc['frames'],folds,groups,strict=True)):
        if fold==0 or not frame['panel'].startswith('engdes2-'):
            continue
        rows.append(dict(frame,index=index,fold=int(fold),source_group=group))
    if not rows:
        raise ValueError('No eligible fit-only indoor frames')
    output.mkdir(parents=True)
    for start in range(0,len(rows),9):
        sheet=Image.new('RGB',(960,810),'white')
        draw=ImageDraw.Draw(sheet)
        for offset,row in enumerate(rows[start:start+9]):
            x,y=offset%3*320,offset//3*270
            with Image.open(root/row['path']) as image:
                sheet.paste(image.convert('RGB'),(x,y+30))
            draw.text((x+3,y+2),f"frame {row['index']} / {Path(row['source']).name[:25]}",fill='black')
            draw.text((x+3,y+15),f"reviewed boxes: {len(row['truth'])}",fill='black')
            for target in row['truth']:
                a,b,c,d=target['box']
                draw.rectangle((x+a,y+30+b,x+c,y+30+d),outline='lime',width=1)
        sheet.save(output/f'page-{start//9:02d}.png')
    queue=dict(frames=rows,source_manifest=str(root/'manifest.json'),
        source_manifest_sha256=sha256(root/'manifest.json'),script_sha256=sha256(__file__),
        sheets={p.name:sha256(p) for p in sorted(output.glob('page-*.png'))},
        scope='Fit-only indoor images, existing reviewed boxes; no model predictions. Completeness not yet approved.',
        review_required=True,training_approved=False,deployment_approved=False,test_evaluated=False)
    write_json(output/'review.json',queue)
    return dict(frames=len(rows),pages=len(queue['sheets']),queue_sha256=sha256(output/'review.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(prepare(a.data,a.output))
