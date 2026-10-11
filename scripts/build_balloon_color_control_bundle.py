"""Stage matched color-control replay, leaving the active Pi deployment untouched."""
import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from dtr.runtime import Predictor
from dtr.vision import RED_BLUE_PROFILE, validate_model_profile
from scripts.pi_balloon_search_bench import check_predictions, golden_key, process_frame
from scripts.pi_red_blue_bench import verify_bundle

ASPECT = 2.42913421076193
SCRIPTS = ('pi_balloon_search_bench.py', 'pi_red_blue_bench.py',
           'research_balloon_search.py', 'research_balloon_search_fast.py')


def prepare(reference, output, library, model):
    reference, output, model = Path(reference), Path(output), Path(model)
    if output.exists():
        raise FileExistsError(output)
    receipt = verify_bundle(reference)
    inputs = read_json(reference/'inputs.json')
    meta = read_json(model.with_suffix('.json'))
    if (receipt.get('deployment_approved') is not False or receipt.get('test_evaluated') is not False
            or not inputs.get('split', '').startswith('development')
            or meta.get('deployment_approved') is not False
            or meta.get('sha256') != sha256(model)
            or meta.get('training', {}).get('mode') != 'control'):
        raise ValueError('Require verified research control and development replay sources')
    validate_model_profile(meta, RED_BLUE_PROFILE)
    name = inputs['default_model']
    alias = 'fragment-color-47'
    if alias in inputs['models'] or meta['training']['seed'] != 47:
        raise ValueError('Unexpected candidate alias or seed')
    key = golden_key('mser', 'split_merged', ASPECT)
    previous = read_json(reference/'search-golden.json')[name][key]
    if not inputs['frames'] or len(previous) != len(inputs['frames']):
        raise ValueError('Incomplete reference golden coverage')
    for relative in receipt['files']:
        dest = output/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference/relative, dest)
    # Freeze the code used for local parity, not an older reference's runtime.
    for source in Path('src/dtr').rglob('*'):
        if source.is_file() and source.suffix in ('.py', '.c', '.h'):
            dest = output/'dtr'/source.relative_to('src/dtr')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    for script in SCRIPTS:
        shutil.copyfile(Path('scripts')/script, output/script)
    relative = f'models/{alias}.tflite'
    shutil.copyfile(model, output/relative)
    shutil.copyfile(model.with_suffix('.json'), (output/relative).with_suffix('.json'))
    inputs['models'][alias] = dict(path=relative)
    # Keep incumbent as default; selection is always explicit in replay CLI.
    native = NativeBalloonRegions(library, direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256(output/'dtr/native_balloon_regions.c'):
        raise ValueError('Native source differs')
    goldens = {}
    for alias_to_test in (name, alias):
        predictor = Predictor(output/inputs['models'][alias_to_test]['path'], allow_unvalidated=True)
        inputs['models'][alias_to_test]['golden'] = [dict(path=row['path'], **{
            k: v for k, v in predictor.predict(output/row['path']).items()
            if k in ('scores', 'label', 'accepted')}) for row in inputs['crops']]
        inputs['models'][alias_to_test]['score_tolerance'] = .001
        records = []
        for row, old in zip(inputs['frames'], previous, strict=True):
            with Image.open(output/row['path']) as image:
                rgb = np.asarray(image.convert('RGB'))
            plain = process_frame(rgb, predictor, 'mser', selection='split_merged', split_max_aspect=ASPECT)
            fast = process_frame(rgb, predictor, 'mser_direct', native, 'split_merged', ASPECT)
            check_predictions(plain['detections'], fast['detections'])
            if alias_to_test == name:
                check_predictions(plain['detections'], old['detections'])
            records.append(dict(source=row['source'], detections=plain['detections']))
        goldens[alias_to_test] = {key: records}
    write_json(output/'inputs.json', inputs)
    write_json(output/'search-golden.json', goldens)
    manifest = dict(files={str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
                          if p.is_file() and p != output/'bundle.json'},
        reference_bundle_sha256=sha256(reference/'bundle.json'), builder_sha256=sha256(__file__),
        candidate_model_sha256=sha256(model), models_compared=[name, alias],
        selection='split_merged', split_max_aspect=ASPECT,
        scope='Research replay only; explicit one-model-per-process comparison; no camera or actuation',
        deployment_approved=False, test_evaluated=False, pi_tested=False)
    write_json(output/'bundle.json', manifest)
    verify_bundle(output)
    return dict(path=str(output), sha256=sha256(output/'bundle.json'), frames=len(previous))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('reference', 'output', 'library', 'model'):
        parser.add_argument('--'+arg, required=True)
    args = parser.parse_args()
    print(prepare(args.reference, args.output, args.library, args.model))
