"""Cross full MSER coverage with direct detections on fixed TRAIN panels."""
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.audit_scale_balanced_search import checked_panels
from scripts.audit_joint_balanced_balloon import checked_receipt, PANELS
from scripts.audit_balanced_localizer import screen
from scripts.audit_source_aware_balloon import aggregate, with_predictions
from scripts.research_blue_bright_search import search
from scripts.research_confident_containment import select as select_crops
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.direct_hybrid_balloon import merge

PROTOCOL = Path('configs/direct-hybrid-balloon-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/direct-hybrid-balloon-20261011.json')
RAW = Path('runs/direct-hybrid-balloon-scenes-20261011.json')
LABELS = ('background', 'red_balloon', 'blue_balloon')


def paired_gate(panels, policy):
    if set(panels) != set(PANELS) or policy not in ('pooled', 'base_first'):
        raise ValueError('Incomplete panels or unknown policy')
    return all(all(p[f'balanced_admit{s}_{policy}']['no_regression'] for p in panels.values()) and
        any(p[f'balanced_admit{s}_{policy}']['effect']['gained_targets'] for p in panels.values())
        for s in (47, 48))


def checked_inputs():
    p = read_json(PROTOCOL)
    if (p['policies'] != ['pooled', 'base_first'] or
            [p[k] for k in ('maximum_crop_calls', 'maximum_direct_candidates',
                'maximum_combined_candidates', 'crop_threshold', 'direct_threshold')] != [12, 12, 24, .8, .5]):
        raise ValueError('Fixed experiment changed')
    reports = {}
    for name in ('direct', 'joint'):
        path = p[f'{name}_audit']
        if sha256(path) != p[f'{name}_audit_sha256']:
            raise ValueError('Audited input changed')
        r = read_json(path)
        for key in ('training_performed', 'development_evaluated', 'test_evaluated',
                    'pi_tested', 'deployment_approved', 'qualification_achieved'):
            if r[key] is not False or p[key] is not False:
                raise ValueError('Require unpromoted TRAIN-only inputs')
        for source, digest in {**r['inputs_sha256'], **r.get('models_sha256', {})}.items():
            if sha256(source) != digest:
                raise ValueError('Audited source or model changed')
        if sha256(r['raw']['path']) != r['raw']['sha256']:
            raise ValueError('Audited detections changed')
        reports[name] = r
    joint = read_json('configs/joint-balanced-balloon-20261011.json')
    checked_receipt(joint['localizer_audit'], joint['localizer_audit_sha256'], 'training')
    classifiers = checked_receipt(joint['classifier_audit'], joint['classifier_audit_sha256'], 'results')
    return p, reports, joint, classifiers


def raw_direct(frame):
    # Reset all previous selection state; keep scores/geometry/acceptance intact.
    return [dict({k: row[k] for k in ('box', 'crop_box', 'area', 'color_group', 'proposal_score',
                                    'label', 'score', 'scores')}, accepted=row['raw_accepted'])
            for row in frame['detections']]


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve hybrid experiment')
    p, reports, joint, classifiers = checked_inputs()
    direct = read_json(reports['direct']['raw']['path'])
    previous = read_json(reports['joint']['raw']['path'])
    panels = checked_panels()
    if set(panels) != set(PANELS) or set(direct) != set(PANELS):
        raise ValueError('Unexpected TRAIN panels')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    paths = {'parent': Path(joint['parent_classifier'])}
    paths.update({name: Path(r['run'])/'student.keras' for name, r in classifiers['results'].items()})
    models = {name: keras.models.load_model(path, compile=False) for name, path in paths.items()}
    if set(models) != {'parent', 'control47', 'control48', 'admit47', 'admit48'} or any(
            m.count_params() != 7763 or m.get_config() != models['parent'].get_config() for m in models.values()):
        raise ValueError('Classifier controls or architecture changed')
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source changed')
    raw, summaries = {}, {}
    for panel, sources in panels.items():
        candidates, crops = [], []
        for _, rgb in sources:
            found = search(rgb, native, True)
            if len(found) > 12:
                raise ValueError('Crop inference budget exceeded')
            candidates.append(found)
            for row in found:
                a, b, c, d = row['crop_box']
                crops.append(np.asarray(Image.fromarray(rgb[b:d, a:c]).resize(
                    (64, 64), Image.Resampling.BILINEAR), np.float32))
        x = np.stack(crops)
        scores = {name: softmax(np.concatenate([m(x[i:i+64], training=False).numpy()
                  for i in range(0, len(x), 64)])) for name, m in models.items()}
        full = {f'{localizer}_{classifier}{seed}_{policy}': []
            for localizer in ('parent', 'control', 'balanced') for classifier in ('parent', 'control', 'admit')
            for seed in (47, 48) for policy in p['policies']}
        full.update({f'base_{name}': [] for name in models})
        offset = 0
        for index, (frame, _) in enumerate(sources):
            base = {}
            for name in models:
                found = []
                for proposal, probabilities in zip(candidates[index], scores[name][offset:offset+len(candidates[index])], strict=True):
                    label = int(probabilities.argmax())
                    found.append(dict(proposal, label=LABELS[label], score=float(probabilities[label]),
                        scores=probabilities.tolist(), accepted=bool(label > 0 and probabilities[label] >= .8)))
                base[name] = found
                result = with_predictions(frame, select_crops(found))
                result['full_frame_neural_calls'] = 0
                full[f'base_{name}'].append(result)
            offset += len(candidates[index])
            for name in (n for n in full if not n.startswith('base_')):
                localizer, classifier_seed, policy = name.split('_', 2)
                seed = classifier_seed[-2:]
                key = localizer+seed if localizer != 'parent' else 'parent'
                f = direct[panel][key]['frames'][index]
                if any(f[k] != frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')):
                    raise ValueError('Direct source identity/order changed')
                classifier = 'parent' if classifier_seed.startswith('parent') else classifier_seed
                detections = merge(base[classifier], raw_direct(f), policy)
                result = with_predictions(frame, detections)
                result.update(neural_calls=len(base[classifier]), full_frame_neural_calls=1)
                full[name].append(result)
        if offset != len(x):
            raise ValueError('Crop accounting mismatch')
        raw[panel] = {name: aggregate(frames) for name, frames in full.items()}
        baseline = previous[panel]['parent_parent47_current']
        summaries[panel] = {name: screen(result, baseline) for name, result in raw[panel].items()}
        for seed in (47, 48):
            for policy in p['policies']:
                name = f'balanced_admit{seed}_{policy}'
                summaries[panel][name]['comparisons'] = {
                    'matched_control': difference(raw[panel][name], raw[panel][f'control_control{seed}_{policy}']),
                    'classifier_only': difference(raw[panel][name], raw[panel][f'base_admit{seed}']),
                    'localizer_only': difference(raw[panel][name], raw[panel][f'balanced_parent{seed}_{policy}'])}
        print(dict(panel=panel, candidates={n: r['metrics'] for n, r in summaries[panel].items()
                                           if n.startswith('balanced_admit')}), flush=True)
    gates = {policy: paired_gate(summaries, policy) for policy in p['policies']}
    checked_inputs()
    inputs = [PROTOCOL, __file__, *[p[f'{n}_audit'] for n in ('direct', 'joint')],
        'scripts/direct_hybrid_balloon.py', 'scripts/direct_localizer_balloon.py',
        'scripts/source_aware_balloon_selection.py', 'scripts/research_blue_bright_search.py',
        'scripts/research_confident_containment.py', 'scripts/audit_source_aware_balloon.py', native.path]
    write_json(RAW, raw)
    write_json(OUTPUT, dict(panels=summaries, eligible_for_separate_export_development=gates,
        raw=dict(path=str(RAW), sha256=sha256(RAW)), inputs_sha256={str(path): sha256(path) for path in inputs},
        models_sha256={str(path): sha256(path) for path in paths.values()},
        decision='Separate export/DEV investigation permitted for passing paired policy.' if any(gates.values()) else
                 'Reject direct hybrid: paired TRAIN retention and recovery gates failed.',
        scope='41variants on82TRAIN scenes. Fresh classification of all bounded MSER candidates plus hash-verified direct detections. One full-frame localizer, at most12crop inferences and24combined candidates per hybrid frame. Baseline MSER-only controls use no localizer. No crop classification for neural boxes and no on-device speed claim.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates), flush=True)


if __name__ == '__main__':
    audit()
