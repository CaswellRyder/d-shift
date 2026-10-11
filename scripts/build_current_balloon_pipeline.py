"""Source-verified latest research model replay, isolated from deployed Pi bundle."""
import inspect
from pathlib import Path
import shutil

from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.audit_balloon_dense_proposals import decode, supplement
from scripts.compare_current_blue_search import MODEL, MODEL_SHA
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.prepare_indoor5_runtime_review import SEARCH, SEARCH_SHA
from scripts.replay_balloon_suppression import raw
from scripts.research_balloon_search_fast import search_fast

OUTPUT = Path('runs/current-balloon-pipeline-pi-20261011')


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report_path = Path('research/balloon-red-blue-20261008/balloon-confidence-selection-development-20261011.json')
    report = read_json(report_path)
    raw_path = Path(report['raw']['path'])
    if (sha256(MODEL) != MODEL_SHA or sha256(SEARCH) != SEARCH_SHA
            or sha256(raw_path) != report['raw']['sha256']
            or any(sha256(p) != h for p, h in report['helpers_sha256'].items())):
        raise ValueError('Frozen model or evaluation changed')
    native_path = Path('artifacts/balloon-regions-mac-20261008/regions.so')
    native = NativeBalloonRegions(native_path, direct=True)
    if read_json(native_path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Host native source changed')
    OUTPUT.mkdir()
    for directory in ('images', 'models', 'scripts'):
        (OUTPUT/directory).mkdir()
    frames, inputs = [], {str(report_path): sha256(report_path), str(raw_path): sha256(raw_path)}
    for panel, variants in read_json(raw_path).items():
        source, review = reviewed_frames('data/'+panel, 'configs/'+panel+'.json')
        inputs['configs/'+panel+'.json'] = sha256('configs/'+panel+'.json')
        old = variants['parent/confidence']['frames']
        for (r, rgb), frame in zip(source, old, strict=True):
            if (r['id'] != frame['id'] or r['source'] != frame['source']
                    or r['sha256'] != frame['frame_sha256'] or r['provisional_truth'] != frame['truth']):
                raise ValueError('Scene identity changed')
            fast = search_fast(rgb, native)
            found = raw(frame['detections'])
            if not len(fast) <= len(found) <= min(12, len(fast)+2):
                raise ValueError('Proposal budget changed')
            for a, b in zip(fast, found):
                if any(a[k] != b[k] for k in ('box', 'crop_box', 'area', 'color_group', 'proposal_score')):
                    raise ValueError('Native source proposals differ')
            path = f'images/{len(frames):03d}.png'
            Image.fromarray(rgb).save(OUTPUT/path)
            frames.append(dict(path=path, panel=panel, **{k: frame[k] for k in
                ('id', 'source', 'frame_sha256', 'truth')}, expected=frame['detections']))
    if len(frames) != 141:
        raise ValueError('Require complete three-panel replay')
    write_json(OUTPUT/'frames.json', frames)
    for source, dest in ((MODEL, 'classifier'), (SEARCH, 'localizer')):
        shutil.copyfile(source, OUTPUT/f'models/{dest}.tflite')
        shutil.copyfile(source.with_suffix('.json'), OUTPUT/f'models/{dest}.json')
    for source in Path('src/dtr').rglob('*'):
        if source.is_file() and source.suffix in ('.py', '.c', '.h'):
            target = OUTPUT/'dtr'/source.relative_to('src/dtr')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name in ('pi_current_balloon_pipeline', 'pi_balloon_search_bench', 'pi_red_blue_bench',
                 'research_balloon_search', 'research_balloon_search_fast', 'research_balloon_confidence_selection'):
        shutil.copyfile(f'scripts/{name}.py', OUTPUT/f'scripts/{name}.py')
    (OUTPUT/'scripts/current_dense_frozen.py').write_text(
        'import cv2\nimport numpy as np\nfrom dtr.tracking import iou\n\n'+inspect.getsource(decode)+'\n'+inspect.getsource(supplement))
    write_json(OUTPUT/'bundle.json', dict(files={str(p.relative_to(OUTPUT)): sha256(p)
        for p in OUTPUT.rglob('*') if p.is_file()}, source_reports=inputs,
        builder_sha256=sha256(__file__), decoder_original_sha256=sha256('scripts/audit_balloon_dense_proposals.py'),
        frames=141, classifier_sha256=MODEL_SHA, localizer_sha256=SEARCH_SHA,
        deployment_approved=False, test_evaluated=False,
        scope='Research classifier plus coarse47 blue proposals plus confidence selection. Complete development replay, not live qualification.'))
    print(dict(path=str(OUTPUT), frames=141, sha256=sha256(OUTPUT/'bundle.json')))


if __name__ == '__main__':
    build()
