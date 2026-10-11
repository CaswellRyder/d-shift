"""Cross audited localization and classification interventions on TRAIN scenes only."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.audit_scale_balanced_search import checked_panels
from scripts.audit_balanced_localizer import screen
from scripts.diagnose_clutter_stage_attribution import crops, difference
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent

PROTOCOL = Path('configs/joint-balanced-balloon-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/joint-balanced-balloon-20261011.json')
RAW = Path('runs/joint-balanced-balloon-scenes-20261011.json')
PANELS = {'original': 64, 'window1': 6, 'window2': 12}


def paired_gate(panels, policy):
    if set(panels) != set(PANELS) or policy not in ('current', 'reserved'):
        raise ValueError('Incomplete panel coverage or undeclared policy')
    return all(all(p[f'balanced_admit{seed}_{policy}']['no_regression'] for p in panels.values())
               and any(p[f'balanced_admit{seed}_{policy}']['effect']['gained_targets'] for p in panels.values())
               for seed in (47, 48))


def checked_receipt(path, expected, records_key):
    if sha256(path) != expected:
        raise ValueError('Experiment receipt changed')
    doc = read_json(path)
    for key in ('development_evaluated', 'test_evaluated', 'pi_tested',
                'deployment_approved', 'qualification_achieved'):
        if doc[key] is not False:
            raise ValueError('Require unpromoted TRAIN-only inputs')
    raw = doc['raw']
    if sha256(raw['path']) != raw['sha256']:
        raise ValueError('Audited predictions changed')
    for record in doc[records_key].values():
        root = Path(record['run'])
        for name, digest in record['artifacts_sha256'].items():
            if sha256(root/name) != digest:
                raise ValueError('Audited checkpoint changed')
        if read_json(root/'status.json')['state'] != 'complete':
            raise ValueError('Training is not complete')
        for name, digest in read_json(root/'provenance.json')['sources_sha256'].items():
            if sha256(name) != digest:
                raise ValueError('Training source changed')
    return doc


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve composition evidence')
    p = read_json(PROTOCOL)
    if (p['seeds'] != [47, 48] or p['policies'] != ['current', 'reserved']
            or p['panels'] != PANELS or p['threshold'] != .8 or p['maximum_candidates'] != 12
            or any(p[k] is not False for k in ('training_performed', 'development_evaluated',
                'test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Fixed composition experiment changed')
    loc = checked_receipt(p['localizer_audit'], p['localizer_audit_sha256'], 'training')
    cla = checked_receipt(p['classifier_audit'], p['classifier_audit_sha256'], 'results')
    for path, digest in loc['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Proposal generation source changed')
    if sha256(p['parent_classifier']) != p['parent_classifier_sha256']:
        raise ValueError('Parent classifier changed')
    geometry = read_json(loc['raw']['path'])
    panels = checked_panels()
    if set(geometry) != set(PANELS) or set(panels) != set(PANELS):
        raise ValueError('Unexpected input panels')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    paths = {'parent': Path(p['parent_classifier'])}
    paths.update({name: Path(r['run'])/'student.keras' for name, r in cla['results'].items()})
    if set(paths) != {'parent', 'control47', 'control48', 'admit47', 'admit48'}:
        raise ValueError('Expected both paired classifier seeds')
    models = {name: keras.models.load_model(path, compile=False) for name, path in paths.items()}
    if any(m.count_params() != 7763 or m.get_config() != models['parent'].get_config() for m in models.values()):
        raise ValueError('Classifier architecture changed')
    raw, summaries = {}, {}
    for panel, sources in panels.items():
        pixels = {f['source']: (f['frame_sha256'], f['truth'], rgb) for f, rgb in sources}
        if len(pixels) != PANELS[panel]:
            raise ValueError('Incomplete or duplicated TRAIN scenes')
        results = {}
        for seed in (47, 48):
            for localizer in ('parent', 'control', 'balanced'):
                for policy in p['policies']:
                    key = f'{localizer}{seed}_{policy}' if localizer != 'parent' else f'parent_{policy}'
                    frames = geometry[panel][key]['frames']
                    if len(frames) != PANELS[panel] or any(len(f['detections']) > 12 for f in frames):
                        raise ValueError('Proposal budget or frame coverage changed')
                    x = crops(frames, pixels)
                    for classifier in ('parent', 'control', 'admit'):
                        model = models['parent' if classifier == 'parent' else f'{classifier}{seed}']
                        logits = np.concatenate([model(x[i:i+64], training=False).numpy()
                                                 for i in range(0, len(x), 64)])
                        result = replay(frames, softmax(logits))
                        if classifier == 'parent':
                            for actual, expected in zip(result['frames'], frames, strict=True):
                                check_replayed_parent(actual, expected)
                        results[f'{localizer}_{classifier}{seed}_{policy}'] = result
        raw[panel] = results
        summaries[panel] = {name: screen(result, geometry[panel]['parent_current'])
                            for name, result in results.items()}
        for seed in (47, 48):
            for policy in p['policies']:
                name = f'balanced_admit{seed}_{policy}'
                summaries[panel][name]['comparisons'] = {
                    'full_matched_control': difference(results[name], results[f'control_control{seed}_{policy}']),
                    'localizer_only': difference(results[name], results[f'balanced_parent{seed}_{policy}']),
                    'classifier_only': difference(results[name], results[f'parent_admit{seed}_{policy}'])}
        print(dict(panel=panel, candidates={name: row['metrics'] for name, row in summaries[panel].items()
                                           if name.startswith('balanced_admit')}), flush=True)
    gates = {policy: paired_gate(summaries, policy) for policy in p['policies']}
    inputs = [PROTOCOL, __file__, p['localizer_audit'], p['classifier_audit'],
              'scripts/audit_balanced_localizer.py', 'scripts/audit_scale_balanced_search.py',
              'scripts/diagnose_clutter_stage_attribution.py', 'scripts/diagnose_clutter_round3_feature.py',
              'scripts/research_confident_containment.py', 'scripts/summarize_public_balloon_trial.py']
    write_json(RAW, raw)
    write_json(OUTPUT, dict(panels=summaries, eligible_for_separate_export_development=gates,
        raw=dict(path=str(RAW), sha256=sha256(RAW)),
        inputs_sha256={str(path): sha256(path) for path in inputs},
        models_sha256={str(path): sha256(path) for path in paths.values()},
        decision='Separate export/DEV investigation permitted for passing paired policy.' if any(gates.values())
                 else 'Reject composition: paired TRAIN target retention and recovery gate failed.',
        scope='Fresh classification of hash-verified proposals on82TRAIN scenes; full factorial paired controls. Maximum12classifications and one2342parameter localizer per frame. No inference ensemble. No new fitting, threshold tuning, export or Pi speed measurement.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates), flush=True)


if __name__ == '__main__':
    audit()
