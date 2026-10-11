"""Reproduce reviewed supervision and matched original-crop TRAIN scene results."""
import csv
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.scene_teacher_supervision import tensors, targets, PROTOCOL, REVIEW
from scripts.clutter_round3_feature_data import hashes
from scripts.balloon_scene_constraints import RAW as TRAIN, RAW_SHA, decisions
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.train_batch3_retraining import calculate

OUTPUT = Path('research/balloon-red-blue-20261008/scene-teacher-supervision-20261011.json')
RAW = Path('runs/scene-teacher-supervision-scenes-20261011.json')


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve supervision audit')
    data, protocol = tensors(), read_json(PROTOCOL)
    p = data['p']
    if sha256(TRAIN) != RAW_SHA or sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Reference changed')
    raw = read_json(TRAIN)
    if set(raw) != {'training'}:
        raise ValueError('TRAIN only')
    frames = raw['training']['parent']['frames']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'], compile=False)
    before = parent(data['x'], training=False).numpy()
    probe_before = parent(data['probes'], training=False).numpy()
    expected_target = hashlib.sha256(targets(data, before).tobytes()).hexdigest()
    original = replay(frames, softmax(before[-385:]))
    for a, b in zip(original['frames'], frames, strict=True):
        check_replayed_parent(a, b)
    results, scenes = {}, dict(parent=original)
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            root = Path(f'runs/clutter-round3-feature-admit-seed{seed}-20261011' if arm == 'control'
                else f'runs/scene-teacher-supervision-reviewed-seed{seed}-20261011')
            prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
            if (read_json(root/'status.json')['state'] != 'complete' or prov['settings'] != p
                    or prov['seed'] != seed or prov['arm'] != ('admit' if arm == 'control' else 'reviewed')
                    or prov['tensor_sha256'] != hashes(data) or prov['sample_identity'] != data['sample_identity']
                    or prov['scene_identities'] != data['scene_ids'] or prov['fresh_optimizer'] is not True
                    or prov['fixed_final_epoch'] is not True or report['optimizer_steps'] != 12560
                    or any(sha256(path) != h for path, h in prov['sources_sha256'].items())
                    or any(prov[k] is not False for k in ('development_evaluated', 'test_evaluated', 'deployment_approved'))):
                raise ValueError('Control or treatment recipe/source mismatch')
            if arm == 'reviewed' and (prov['protocol'] != protocol
                    or prov['correction_labels'] != data['correction_labels'].tolist()
                    or prov['target_sha256'] != expected_target):
                raise ValueError('Reviewed loss targets changed')
            with (root/'epochs.csv').open() as handle:
                trace = list(csv.DictReader(handle))
            if ([int(r['epoch']) for r in trace] != list(range(80))
                    or not all(np.isfinite(float(r['loss'])) for r in trace)):
                raise ValueError('Incomplete training schedule')
            for key, file in (('model_sha256', 'student.keras'), ('predictions_sha256', 'predictions.npz'),
                              ('provenance_sha256', 'provenance.json')):
                if sha256(root/file) != report[key]:
                    raise ValueError('Training artifact changed')
            model = keras.models.load_model(root/'student.keras', compile=False)
            if model.count_params() != 7763 or model.get_config() != parent.get_config():
                raise ValueError('Inference model changed')
            after = model(data['x'], training=False).numpy()
            probes = model(data['probes'], training=False).numpy()
            with np.load(root/'predictions.npz', allow_pickle=False) as stored:
                for key, value in dict(before=before, after=after, labels=data['y'], probe_before=probe_before,
                                       probe_after=probes, probe_labels=data['probe_labels']).items():
                    np.testing.assert_allclose(stored[key], value, atol=1e-5, rtol=0)
            measured = calculate(before[:2074], after[:2074], data['y'][:2074], probes, data['probe_labels'])
            legacy = measured.pop('training_gate_pass')
            changes = np.flatnonzero(decisions(before[-385:]) != decisions(after[-385:])).tolist()
            if any(report[k] != v for k, v in measured.items()) or report['scene_changed_rows'] != changes:
                raise ValueError('Crop diagnostics failed to reproduce')
            if arm == 'reviewed' and (report['legacy_perfect_crop_gate'] != legacy or report['training_gate_pass'] is not False):
                raise ValueError('Provisional gate changed')
            found = replay(frames, softmax(after[-385:]))
            effect = difference(found, original)
            negatives = [false_acceptances(z) for z in (probe_before, probes)]
            mask = data['correction_labels'] >= 0
            corrected = data['correction_labels'][mask]
            predictions = softmax(after[mask])
            review_correct = ((predictions.argmax(1) == corrected)
                & ((corrected == 0) | (predictions.max(1) >= .8))).tolist()
            name = f'{arm}{seed}'
            scenes[name] = found
            results[name] = dict(run=str(root), model_sha256=report['model_sha256'],
                report_sha256=sha256(root/'report.json'), provenance_sha256=sha256(root/'provenance.json'),
                predictions_sha256=sha256(root/'predictions.npz'), epochs_sha256=sha256(root/'epochs.csv'),
                metrics=found['metrics'], effect=effect, background_false_acceptances=negatives,
                review_correct=review_correct, training_scene_gate_pass=scene_gate(effect, *negatives),
                lost_old_crops=len(measured['lost_old_rows']))
            print(dict(name=name, metrics=found['metrics'], lost=len(effect['lost_targets']),
                corrected=sum(review_correct), gate=results[name]['training_scene_gate_pass']), flush=True)
    passed = all(results[f'reviewed{s}']['training_scene_gate_pass'] for s in (47, 48))
    write_json(RAW, scenes)
    write_json(OUTPUT, dict(results=results, protocol_sha256=sha256(PROTOCOL), review_sha256=sha256(REVIEW),
        script_sha256=sha256(__file__), raw=dict(path=str(RAW), sha256=sha256(RAW)),
        eligible_for_export_and_development=passed,
        decision='Eligible for export parity and repeatedDEV, not deployment.' if passed else 'Reject: reviewed pair failed TRAIN scene nonregression.',
        scope='Original-crop matched experiment. Reused controls verified against exact recipe, tensors and source identities. TRAIN scene counts only, no independent qualification.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
