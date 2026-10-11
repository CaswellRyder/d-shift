"""TRAIN-only direct-detector feasibility, fixed threshold and no crop inference."""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.audit_joint_balanced_balloon import checked_receipt, PANELS
from scripts.audit_scale_balanced_search import checked_panels
from scripts.audit_balanced_localizer import screen
from scripts.balloon_dense_runtime import prepare_input, DenseProposals
from scripts.evaluate_scene_decision_retention import LOCALIZER, LOCALIZER_SHA
from scripts.direct_localizer_balloon import detect
from scripts.audit_source_aware_balloon import with_predictions, aggregate

PROTOCOL = Path('configs/direct-localizer-balloon-20261011.json')
OUTPUT = Path('research/balloon-red-blue-20261008/direct-localizer-balloon-20261011.json')
RAW = Path('runs/direct-localizer-balloon-scenes-20261011.json')


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve direct detector evidence')
    p = read_json(PROTOCOL)
    if (p['threshold'] != .5 or p['maximum_candidates'] != 12 or
            any(p[k] is not False for k in ('training_performed', 'development_evaluated',
                'test_evaluated', 'pi_tested', 'deployment_approved', 'qualification_achieved'))):
        raise ValueError('Fixed TRAIN experiment changed')
    previous = checked_receipt(p['localizer_audit'], p['localizer_audit_sha256'], 'training')
    for path, digest in previous['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Localizer source changed')
    if sha256(LOCALIZER) != LOCALIZER_SHA:
        raise ValueError('Parent TFLite changed')
    baseline = read_json(previous['raw']['path'])
    panels = checked_panels()
    if set(panels) != set(PANELS):
        raise ValueError('Unexpected TRAIN panels')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    models = {name: keras.models.load_model(Path(row['run'])/'proposal.keras', compile=False)
              for name, row in previous['training'].items()}
    parent = DenseProposals(LOCALIZER)
    raw, summaries = {}, {}
    for panel, sources in panels.items():
        x = np.concatenate([prepare_input(rgb) for _, rgb in sources])
        predictions = {name: m(x, training=False).numpy() for name, m in models.items()}
        parent_values = []
        for _, rgb in sources:
            parent.predict(rgb)
            parent_values.append(parent.interpreter.get_tensor(parent.output['index'])[0].copy())
        predictions['parent'] = np.stack(parent_values)
        raw[panel], summaries[panel] = {}, {}
        for name, values in predictions.items():
            frames = []
            for (frame, _), output in zip(sources, values, strict=True):
                found = detect(output, p['threshold'])
                if len(found) > 12:
                    raise ValueError('Candidate budget exceeded')
                result = with_predictions(frame, found)
                # This experiment executes no per-candidate classifier.
                result.update(neural_calls=0, full_frame_neural_calls=1)
                frames.append(result)
            result = aggregate(frames)
            raw[panel][name] = result
            summaries[panel][name] = screen(result, baseline[panel]['parent_current'])
        print(dict(panel=panel, metrics={name: row['metrics'] for name, row in summaries[panel].items()}), flush=True)
    eligible = all(all(panel[f'balanced{seed}']['no_regression'] for panel in summaries.values()) and
        any(panel[f'balanced{seed}']['effect']['gained_targets'] for panel in summaries.values()) for seed in (47, 48))
    inputs = [PROTOCOL, __file__, p['localizer_audit'], LOCALIZER, 'scripts/direct_localizer_balloon.py',
              'scripts/research_blue_peak_blend.py', 'scripts/research_confident_containment.py',
              'scripts/audit_source_aware_balloon.py', 'scripts/audit_balanced_localizer.py',
              'scripts/audit_joint_balanced_balloon.py']
    write_json(RAW, raw)
    write_json(OUTPUT, dict(panels=summaries, eligible_for_separate_export_development=eligible,
        raw=dict(path=str(RAW), sha256=sha256(RAW)), inputs_sha256={str(path): sha256(path) for path in inputs},
        decision='Separate export/DEV investigation permitted.' if eligible else
                 'Reject direct detection: proposal heat scores at fixed0.5 fail paired TRAIN retention/recovery.',
        scope='One2342parameter full-frame model and zero crop-classifier calls per frame. Five checkpoints,82TRAIN scenes each. Decoded heat scores are not established calibrated detection probabilities. No timing, held-out accuracy or hardware claim.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=eligible), flush=True)


if __name__ == '__main__':
    audit()
