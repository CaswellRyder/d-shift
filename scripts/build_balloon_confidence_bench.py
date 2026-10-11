"""Build an isolated stdlib-only selection benchmark, with verbatim source functions."""
import ast
from pathlib import Path
import shutil

from dtr.data import read_json, sha256, write_json

ROOT = Path('runs/balloon-confidence-selection-pi-20261011')
RESEARCH = Path('research/balloon-red-blue-20261008')


def extract(path, names):
    source = Path(path).read_text()
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    return '\n\n'.join(ast.get_source_segment(source, nodes[name]) for name in names)+'\n'


def build():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    doc_path = RESEARCH/'balloon-confidence-selection-development-20261011.json'
    doc = read_json(doc_path)
    raw_path = Path(doc['raw']['path'])
    if (sha256(raw_path) != doc['raw']['sha256']
            or any(sha256(p) != h for p, h in doc['helpers_sha256'].items())):
        raise ValueError('Frozen source or policies changed')
    raw = read_json(raw_path)
    rows = []
    for panel, variants in raw.items():
        for model in ('parent', 'background47'):
            for old, new in zip(variants[model+'/current']['frames'], variants[model+'/confidence']['frames'], strict=True):
                if any(old[k] != new[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Compared frames differ')
                from scripts.replay_balloon_suppression import raw as reconstruct
                found = reconstruct(old['detections'])
                rows.append(dict(panel=panel, model=model, id=old['id'], found=found,
                    expected=dict(current=old['detections'], confidence=new['detections'])))
    if len(rows) != 282:
        raise ValueError('Require both localizers on all141 development frames')
    ROOT.mkdir()
    for name in ('dtr', 'scripts'):
        (ROOT/name).mkdir()
        (ROOT/name/'__init__.py').write_text('')
    specifications = {
        'src/dtr/tracking.py': ('dtr/tracking.py', '', ['iou']),
        'src/dtr/vision.py': ('dtr/vision.py', '', ['suppress_duplicates']),
        'scripts/research_balloon_search.py': ('scripts/research_balloon_search.py', 'from dtr.tracking import iou\n',
            ['suppress_confirmed_balloon_parts', 'split_merged_balloon_parents', 'replace_oversized_balloon_parents']),
        'scripts/pi_balloon_search_bench.py': ('scripts/pi_balloon_search_bench.py',
            'import math\nfrom dtr.vision import suppress_duplicates\nfrom scripts.research_balloon_search import suppress_confirmed_balloon_parts, split_merged_balloon_parents, replace_oversized_balloon_parents\n',
            ['validate_selection', 'select_detections'])}
    for source, (target, imports, functions) in specifications.items():
        constants = "SELECTIONS = ('parts', 'split_merged', 'split_oversized')\n" if target.endswith('pi_balloon_search_bench.py') else ''
        (ROOT/target).write_text(imports+constants+extract(source, functions))
    for name in ('research_balloon_confidence_selection.py', 'pi_balloon_confidence_selection_bench.py'):
        shutil.copyfile('scripts/'+name, ROOT/'scripts'/name)
    write_json(ROOT/'inputs.json', rows)
    files = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.rglob('*') if p.is_file()}
    sources = [*specifications, 'scripts/research_balloon_confidence_selection.py',
        'scripts/pi_balloon_confidence_selection_bench.py']
    write_json(ROOT/'bundle.json', dict(files=files, source_sha256={p: sha256(p) for p in sources},
        development_sha256=sha256(doc_path), builder_sha256=sha256(__file__),
        extracted_functions={p: v[2] for p, v in specifications.items()},
        scope='Verbatim selection functions, no images/model loading/camera/actuation; host goldens required.',
        deployment_approved=False))
    print(ROOT)


if __name__ == '__main__':
    build()
