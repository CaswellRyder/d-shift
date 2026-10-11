"""Freeze an isolated paired-search replay with exact source hashes and goldens."""
from pathlib import Path
import shutil

from dtr.data import read_json, sha256, write_json
from scripts.pi_red_blue_bench import verify_bundle

SOURCE = Path('runs/current-balloon-pipeline-pi-20261011')
OUTPUT = Path('runs/blue-bright-hybrid-pi-20261011')


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old_bundle = verify_bundle(SOURCE)
    reference_path = Path('research/balloon-red-blue-20261008/blue-bright-hybrid-development-20261011.json')
    reference = read_json(reference_path)
    raw_path = Path(reference['raw']['path'])
    if (sha256(raw_path) != reference['raw']['sha256']
            or any(sha256(p) != h for p, h in reference['helpers_sha256'].items())
            or reference['classifier_sha256'] != old_bundle['classifier_sha256']
            or reference['localizer_sha256'] != old_bundle['localizer_sha256']):
        raise ValueError('Source identity changed')
    for path in old_bundle['files']:
        target = OUTPUT/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/path, target)
    indexed = {(row['panel'], row['id']): row for row in read_json(SOURCE/'frames.json')}
    rows = []
    for panel, values in read_json(raw_path).items():
        for old, new in zip(values['full']['frames'], values['blue_bright']['frames'], strict=True):
            base = indexed[(panel, old['id'])]
            if any(old[k] != new[k] or old[k] != base[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                raise ValueError('Paired frame mismatch')
            rows.append(dict(base, expected=dict(original=old['detections'], blue_bright=new['detections'])))
    if len(rows) != 141:
        raise ValueError('Incomplete paired frames')
    write_json(OUTPUT/'paired-frames.json', rows)
    for name in ('research_blue_bright_search', 'pi_blue_bright_hybrid'):
        shutil.copyfile('scripts/'+name+'.py', OUTPUT/'scripts'/f'{name}.py')
    write_json(OUTPUT/'bundle.json', dict(files={str(p.relative_to(OUTPUT)): sha256(p)
        for p in OUTPUT.rglob('*') if p.is_file()}, source_bundle_sha256=sha256(SOURCE/'bundle.json'),
        development_sha256=sha256(reference_path), development_raw_sha256=sha256(raw_path),
        builder_sha256=sha256(__file__), classifier_sha256=reference['classifier_sha256'],
        localizer_sha256=reference['localizer_sha256'], scope='Sequential paired research replay only',
        test_evaluated=False, deployment_approved=False))
    print(dict(path=str(OUTPUT), frames=141, sha256=sha256(OUTPUT/'bundle.json')))


if __name__ == '__main__':
    build()
