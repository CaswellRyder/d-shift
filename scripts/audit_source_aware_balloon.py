"""Reproduce proposal provenance and isolate source-aware selection on TRAIN."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.audit_scale_balanced_search import checked_panels
from scripts.audit_joint_balanced_balloon import PANELS, checked_receipt, paired_gate
from scripts.audit_balanced_localizer import screen
from scripts.balloon_dense_runtime import DenseProposals, prepare_input
from scripts.evaluate_scene_decision_retention import LOCALIZER, LOCALIZER_SHA
from scripts.research_blue_bright_search import search
from scripts.research_blue_peak_blend import decode
from scripts.reserved_dense_balloon_search import reserve
from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_confident_containment import select as incumbent_select
from scripts.source_aware_balloon_selection import select
from scripts.diagnose_clutter_round3_feature import check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.evaluate_red_blue_development import detection_counts, metrics, LABELS

PROTOCOL = Path('configs/source-aware-balloon-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/source-aware-balloon-20261011.json')
RAW = Path('runs/source-aware-balloon-scenes-20261011.json')


def checked_inputs():
    p = read_json(PROTOCOL)
    if (sha256(p['source_audit']) != p['source_audit_sha256'] or p['maximum_candidates'] != 12
            or p['threshold'] != .8 or p['panels'] != PANELS):
        raise ValueError('Frozen selection experiment changed')
    report = read_json(p['source_audit'])
    for doc in (p, report):
        if any(doc[k] is not False for k in ('training_performed', 'development_evaluated',
                'test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved')):
            raise ValueError('TRAIN-only unpromoted inputs required')
    for path, digest in {**report['inputs_sha256'], **report['models_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Composition inputs changed')
    if sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Composition predictions changed')
    joint = read_json('configs/joint-balanced-balloon-20261011.json')
    loc = checked_receipt(joint['localizer_audit'], joint['localizer_audit_sha256'], 'training')
    checked_receipt(joint['classifier_audit'], joint['classifier_audit_sha256'], 'results')
    for path, digest in loc['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Localizer evaluation dependency changed')
    return p, report, loc


def raw_predictions(frame, proposals):
    found = []
    for saved, proposed in zip(frame['detections'], proposals, strict=True):
        for key in ('box', 'crop_box', 'area', 'color_group', 'proposal_score'):
            np.testing.assert_allclose(saved[key], proposed[key], atol=1e-5, rtol=0)
        row = {k: saved[k] for k in ('box', 'crop_box', 'area', 'color_group', 'proposal_score',
                                    'label', 'score', 'scores')}
        found.append(dict(row, accepted=saved['raw_accepted'], proposal_origin=proposed['proposal_origin']))
    return found


def with_predictions(frame, detections):
    return dict({k: frame[k] for k in ('id', 'source', 'frame_sha256', 'truth')},
                detections=detections, counts=detection_counts(frame['truth'], detections),
                neural_calls=len(detections))


def aggregate(frames):
    return dict(frames=frames, metrics=metrics({c: {k: sum(f['counts'][c][k] for f in frames)
                    for k in ('tp', 'fp', 'fn')} for c in LABELS}))


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve selector evidence')
    p, previous, loc = checked_inputs()
    saved = read_json(previous['raw']['path'])
    panels = checked_panels()
    if set(saved) != set(PANELS):
        raise ValueError('Unexpected scene panel')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    models = {name: keras.models.load_model(Path(record['run'])/'proposal.keras', compile=False)
              for name, record in loc['training'].items()}
    if sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Parent localizer changed')
    parent = DenseProposals(LOCALIZER)
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    raw, summaries = {}, {}
    for panel, frames in panels.items():
        x = np.concatenate([prepare_input(rgb) for _, rgb in frames])
        predictions = {name: m(x, training=False).numpy() for name, m in models.items()}
        results = {name: [] for name in saved[panel]}
        for index, (frame, rgb) in enumerate(frames):
            base = [dict(r, proposal_origin='mser') for r in search(rgb, native, True)]
            parent.predict(rgb)
            values = {'parent': parent.interpreter.get_tensor(parent.output['index'])[0]}
            values.update({name: output[index] for name, output in predictions.items()})
            proposals = {}
            for name, output in values.items():
                learned = [dict(r, proposal_origin='learned') for r in decode(output, True)]
                proposals[name] = {'current': supplement(base, learned), 'reserved': reserve(base, learned)}
            for name, original in saved[panel].items():
                localizer, classifier_seed, policy = name.split('_')
                key = localizer + classifier_seed[-2:] if localizer != 'parent' else 'parent'
                f = original['frames'][index]
                if any(f[k] != frame[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                    raise ValueError('Scene identity or order changed')
                inputs = raw_predictions(f, proposals[key][policy])
                reproduced = with_predictions(f, incumbent_select(inputs))
                check_replayed_parent(reproduced, f)
                results[name].append(with_predictions(f, select(inputs)))
        raw[panel] = {name: aggregate(fs) for name, fs in results.items()}
        baseline = saved[panel]['parent_parent47_current']
        summaries[panel] = {name: dict(screen(result, baseline),
            selection_effect=difference(result, saved[panel][name])) for name, result in raw[panel].items()}
        print(dict(panel=panel, candidates={name: row['metrics'] for name, row in summaries[panel].items()
                                           if name.startswith('balanced_admit')}), flush=True)
    gates = {policy: paired_gate(summaries, policy) for policy in ('current', 'reserved')}
    inputs = [PROTOCOL, __file__, p['source_audit'], 'scripts/source_aware_balloon_selection.py',
        'scripts/research_balloon_confidence_selection.py', 'scripts/compare_balloon_containment.py',
        'scripts/research_balloon_search.py', 'scripts/research_confident_containment.py',
        'scripts/research_blue_bright_search.py', 'scripts/research_blue_peak_blend.py',
        'scripts/reserved_dense_balloon_search.py', 'scripts/audit_balloon_dense_proposals.py', native.path]
    checked_inputs()
    write_json(RAW, raw)
    write_json(OUTPUT, dict(panels=summaries, eligible_for_separate_export_development=gates,
        raw=dict(path=str(RAW), sha256=sha256(RAW)), inputs_sha256={str(path): sha256(path) for path in inputs},
        source_receipt_sha256=sha256(p['source_audit']),
        decision='Separate export/DEV investigation permitted for passing paired policy.' if any(gates.values())
                 else 'Reject: source-priority selection does not meet paired TRAIN retention and recovery.',
        scope='36factorial variants on82TRAIN scenes. Fresh exact proposal reconstruction verifies source tags; every original decision reproduced before post-selection-only replay. Frozen classifier scores and at most12classifications. No held-out accuracy, export or Pi latency claim.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates), flush=True)


if __name__ == '__main__':
    audit()
