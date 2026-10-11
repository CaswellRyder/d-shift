"""Freeze complete development replay without changing active deployment."""
import argparse
import inspect
from pathlib import Path
import shutil

from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from scripts.audit_balloon_dense_proposals import decode
from scripts.balloon_dense_runtime import DenseProposals
from scripts.evaluate_reviewed_balloon_scenes import reviewed_frames
from scripts.pi_dense_selection_replay import GuardedDense, process, compare
from scripts.prepare_dense_compatibility_probe import PANELS


def decoder_source():
    return 'import cv2\nimport numpy as np\nfrom dtr.tracking import iou\n\n'+inspect.getsource(decode)


def build(output, library):
    output = Path(output)
    output.mkdir(exist_ok=False)
    native = NativeBalloonRegions(library, direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Host native source mismatch')
    classifier = Path('runs/balloon-fragment-control-seed47-20261011/color.fp32.tflite')
    predictor = Predictor(classifier, allow_unvalidated=True)
    (output/'models').mkdir()
    (output/'images').mkdir()
    for source, destination in ((classifier,'classifier.tflite'),(classifier.with_suffix('.json'),'classifier.json')):
        shutil.copyfile(source, output/'models'/destination)
    nets, identities = {}, {}
    for seed in ('47','48'):
        path = Path(f'runs/balloon-dense-overlap-l1_giou-seed{seed}-20261011/proposal.fp32.tflite')
        nets[seed] = GuardedDense(DenseProposals(path))
        identities[seed] = sha256(path)
        shutil.copyfile(path, output/f'models/dense-{seed}.tflite')
        shutil.copyfile(path.with_suffix('.json'), output/f'models/dense-{seed}.json')
    records, source_reports = [], {}
    for panel in PANELS:
        path = Path(f'runs/balloon-dense-supported-gated-{panel}-20261011.json')
        report = read_json(path)
        source_reports[str(path)] = sha256(path)
        if (report['classifier_sha256'] != sha256(classifier)
                or report['policy_sha256'] != sha256('scripts/research_dense_supported_selection.py')
                or report['queue_sha256'] != sha256(f'data/{panel}/review.json')
                or report['review_sha256'] != sha256(f'configs/{panel}.json')
                or any(report['dense_models'][s]['sha256'] != identities[s] for s in nets)):
            raise ValueError('Frozen report identities changed')
        expected = {s:{r['id']:r for r in report['results']['supported'+s]['frames']} for s in nets}
        baseline = {r['id']:r for r in report['results']['baseline']['frames']}
        frames, _ = reviewed_frames('data/'+panel, 'configs/'+panel+'.json')
        if set(baseline) != {r['id'] for r,_ in frames}:
            raise ValueError('Incomplete frozen frames')
        for row, rgb in frames:
            for seed, net in nets.items():
                actual = process(rgb, predictor, net, native)
                ref = expected[seed][row['id']]
                compare(actual['detections'], ref['detections'])
                compare(actual['baseline'], baseline[row['id']]['detections'])
                if actual['dense_calls'] != ref['dense_calls'] or actual['dense_error']:
                    raise ValueError('Host native gate mismatch')
            relative = f'images/{len(records):03d}.png'
            Image.fromarray(rgb).save(output/relative)
            records.append(dict(path=relative, panel=panel, id=row['id'],
                baseline=baseline[row['id']]['detections'], dense_calls=ref['dense_calls'],
                selected={s:expected[s][row['id']]['detections'] for s in nets}))
    write_json(output/'frames.json', records)
    for source in Path('src/dtr').rglob('*'):
        if source.is_file() and source.suffix in ('.py','.c','.h'):
            dest = output/'dtr'/source.relative_to('src/dtr')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    (output/'scripts').mkdir()
    for name in ('pi_dense_selection_replay','pi_balloon_search_bench','pi_red_blue_bench',
                 'research_balloon_search','research_balloon_search_fast','research_dense_supported_selection'):
        shutil.copyfile(f'scripts/{name}.py', output/f'scripts/{name}.py')
    # Copy the function verbatim, avoiding all training-data/audit imports on Pi.
    (output/'scripts/dense_decode_frozen.py').write_text(decoder_source())
    write_json(output/'bundle.json',dict(
        files={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()},
        decoder_original_sha256=sha256('scripts/audit_balloon_dense_proposals.py'),
        builder_sha256=sha256(__file__), source_reports=source_reports,
        frames=len(records), dense_calls_per_seed=sum(r['dense_calls'] for r in records),
        scope='Full-pipeline repeated development replay only; no training or final qualification',
        deployment_approved=False))
    return dict(frames=len(records), dense_calls_per_seed=sum(r['dense_calls'] for r in records),
                bundle_sha256=sha256(output/'bundle.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--library', required=True)
    a = p.parse_args()
    print(build(a.output,a.library))
