"""Stage three eligible research arms with exact scene goldens and frozen sources."""
from pathlib import Path
import shutil

from dtr.data import read_json,sha256,write_json
from scripts.audit_indoor5_outcomes import compare_frames
from scripts.evaluate_blue_peak_blend import OUTPUT as SCENES,RAW
from scripts.evaluate_blue_scene_localizer import nonregression
from scripts.pi_red_blue_bench import verify_bundle
from scripts.summarize_public_balloon_trial import checked_metrics

SOURCE = Path('runs/blue-bright-hybrid-pi-20261011')
OUTPUT = Path('runs/blue-peak-blend-pi-20261011')
ARMS = {'original':'parent/original','blend':'parent/blend','head_blend':'background/blend'}


def checked_scenes():
    doc = read_json(SCENES)
    if (sha256(RAW) != doc['raw']['sha256'] or doc['script_sha256'] != sha256('scripts/evaluate_blue_peak_blend.py')
            or doc['protocol_sha256'] != sha256('configs/blue-peak-blend-20261011.json')
            or any(sha256(p) != h for p,h in doc['helpers_sha256'].items())
            or any(sha256(p) != h for p,h in doc['inputs'].items())):
        raise ValueError('Host comparison changed')
    raw = read_json(RAW)
    for panel,variants in raw.items():
        for name,result in variants.items():
            saved = doc['panels'][panel][name]
            compared = compare_frames(result['frames'],variants['parent/original']['frames'])
            if (checked_metrics(result) != saved['metrics'] or any(saved[k] != v for k,v in compared.items())
                    or saved['nonregression'] != nonregression(variants['parent/original']['metrics'],result['metrics'])):
                raise ValueError('Counts or identity retention do not reproduce')
            if name in ('parent/blend','background/blend') and (not saved['nonregression'] or saved['lost_parent_targets']):
                raise ValueError('Candidate not eligible for Pi comparison')
    return doc,raw


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc,raw = checked_scenes()
    source = verify_bundle(SOURCE)
    if (source['classifier_sha256'] != doc['classifier_sha256']
            or source['localizer_sha256'] != doc['models']['parent']['sha256']):
        raise ValueError('Source bundle models differ')
    for path in source['files']:
        target = OUTPUT/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/path,target)
    indexed = {(r['panel'],r['id']):r for r in read_json(SOURCE/'paired-frames.json')}
    rows = []
    for panel,variants in raw.items():
        if panel == 'training':
            continue
        for i,frame in enumerate(variants['parent/original']['frames']):
            source_row = indexed[(panel,frame['id'])]
            expected = {}
            for policy,name in ARMS.items():
                row = variants[name]['frames'][i]
                if any(row[k] != frame[k] or row[k] != source_row[k] for k in ('id','source','frame_sha256','truth')):
                    raise ValueError('Mismatched replay scene')
                expected[policy] = row['detections']
            rows.append(dict(source_row,expected=expected))
    if len(rows) != 141:
        raise ValueError('Incomplete replay')
    write_json(OUTPUT/'decoder-frames.json',rows)
    head = Path(doc['models']['background']['path'])
    if (sha256(head) != doc['models']['background']['sha256']
            or sha256(head.with_suffix('.json')) != doc['models']['background']['metadata_sha256']):
        raise ValueError('Head model changed')
    shutil.copyfile(head,OUTPUT/'models/head_localizer.tflite')
    shutil.copyfile(head.with_suffix('.json'),OUTPUT/'models/head_localizer.json')
    for name in ('research_blue_peak_blend','pi_blue_peak_blend'):
        shutil.copyfile('scripts/'+name+'.py',OUTPUT/'scripts'/f'{name}.py')
    write_json(OUTPUT/'bundle.json',dict(files={str(p.relative_to(OUTPUT)):sha256(p) for p in OUTPUT.rglob('*') if p.is_file()},
        builder_sha256=sha256(__file__),source_bundle_sha256=sha256(SOURCE/'bundle.json'),
        scenes_sha256=sha256(SCENES),scenes_raw_sha256=sha256(RAW),classifier_sha256=doc['classifier_sha256'],
        localizer_sha256=doc['models']['parent']['sha256'],head_localizer_sha256=doc['models']['background']['sha256'],
        arms=ARMS,scope='Isolated sequential three-arm development replay. No camera or actuation.',
        test_evaluated=False,deployment_approved=False))
    print(dict(path=str(OUTPUT),frames=len(rows),sha256=sha256(OUTPUT/'bundle.json')))


if __name__ == '__main__':
    build()
