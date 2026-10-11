"""Reproduce paired TRAIN predictions and independently recount detection retention."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.balloon_hue_trial_data import tensors, PROTOCOL, matched_inputs
from scripts.audit_window1_scene_search import REVIEW as SCENE_REVIEW
from scripts.clutter_round3_feature_data import hashes, targets
from scripts.balloon_scene_constraints import RAW as TRAIN, RAW_SHA, decisions
from scripts.diagnose_clutter_round3_feature import replay, check_replayed_parent
from scripts.diagnose_clutter_stage_attribution import difference
from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances
from scripts.train_batch3_retraining import calculate, correct
from scripts.prepare_openimages_balloon_review import checked_path
from scripts.summarize_public_balloon_trial import checked_metrics

OUTPUT = Path('research/balloon-red-blue-20261008/balloon-hue-20261011.json')
RAW = Path('runs/balloon-hue-scenes-20261011.json')


def training_gate(effect, new_effect, negatives):
    improved = (bool(effect['gained_targets']) or bool(new_effect['gained_targets'])
                or any(v['fp'] < 0 for e in (effect, new_effect) for v in e['count_delta'].values()))
    return bool(scene_gate(effect, *negatives) and scene_gate(new_effect, *negatives) and improved)


def batch_scene_pixels(frames):
    review = read_json(SCENE_REVIEW)
    root = Path(review['queue'])
    if sha256(root/'review.json') != review['queue_sha256']:
        raise ValueError('Scene pixels changed')
    queue = read_json(root/'review.json')
    pixels = []
    for f in frames:
        original = queue['frames'][f['id']]
        if original['sha256'] != f['frame_sha256'] or 'openimages/train/'+original['image_id'] != f['source']:
            raise ValueError('Batch scene identity changed')
        with Image.open(checked_path(root, original['path'], original['sha256'])) as im:
            rgb = im.convert('RGB')
            for d in f['detections']:
                crop = rgb.crop(d['crop_box']).resize((64, 64), Image.Resampling.BILINEAR)
                pixels.append(np.asarray(crop, np.float32))
    return np.stack(pixels)


def verify_run(root, data, arm, seed, parent, before, probe_before):
    p = data['p']
    prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
    expected_target = hashlib.sha256(targets(data['y'], before, data['preserve'], data['scene_mask']).tobytes()).hexdigest()
    if (read_json(root/'status.json')['state'] != 'complete' or prov['settings'] != p
            or prov['protocol'] != data['protocol'] or prov['arm'] != arm or prov['seed'] != seed
            or prov['tensor_sha256'] != hashes(data) or prov['target_sha256'] != expected_target
            or prov['sample_identity'] != data['sample_identity'] or prov['scene_identities'] != data['scene_ids']
            or prov['fresh_optimizer'] is not True or prov['fixed_final_epoch'] is not True
            or report['optimizer_steps'] != p['optimizer_steps']
            or any(sha256(path) != h for path, h in prov['sources_sha256'].items())
            or any(prov[k] is not False for k in ('development_evaluated', 'test_evaluated', 'deployment_approved'))):
        raise ValueError('Run recipe, source or schedule changed')
    with (root/'epochs.csv').open() as handle:
        trace = list(csv.DictReader(handle))
    if [int(r['epoch']) for r in trace] != list(range(80)) or not all(np.isfinite(float(r['loss'])) for r in trace):
        raise ValueError('Incomplete or nonfinite epoch trace')
    for key, file in (('model_sha256', 'student.keras'), ('predictions_sha256', 'predictions.npz'),
                      ('provenance_sha256', 'provenance.json')):
        if sha256(root/file) != report[key]:
            raise ValueError('Training artifact changed')
    import keras
    model = keras.models.load_model(root/'student.keras', compile=False)
    if model.count_params() != 7763 or model.get_config() != parent.get_config():
        raise ValueError('Architecture changed')
    after = model(data['x'], training=False).numpy()
    probes = model(data['probes'], training=False).numpy()
    with np.load(root/'predictions.npz', allow_pickle=False) as stored:
        for key, values in dict(before=before, after=after, labels=data['y'], probe_before=probe_before,
                                probe_after=probes, probe_labels=data['probe_labels']).items():
            np.testing.assert_allclose(stored[key], values, atol=1e-5, rtol=0)
    measured = calculate(before[:2074], after[:2074], data['y'][:2074], probes, data['probe_labels'])
    legacy = measured.pop('training_gate_pass')
    changes = np.flatnonzero(decisions(before[-385:]) != decisions(after[-385:])).tolist()
    if (any(report[k] != v for k, v in measured.items()) or report['scene_changed_rows'] != changes
            or report['legacy_perfect_crop_gate'] != legacy or report['training_gate_pass'] is not False):
        raise ValueError('Crop diagnostics do not reproduce')
    return model, after, probes, measured


def audit():
    if OUTPUT.exists() or RAW.exists():
        raise FileExistsError('Preserve paired audit')
    data = {a:tensors(a) for a in ('control', 'hue')}
    changed = matched_inputs(data['control'], data['hard'])
    if sha256(TRAIN) != RAW_SHA:
        raise ValueError('Original TRAIN geometry changed')
    original_raw = read_json(TRAIN)
    if set(original_raw) != {'training'}:
        raise ValueError('TRAIN scenes only')
    old_frames = original_raw['training']['parent']['frames']
    new_frames = data['control']['batch_frames']
    new_pixels = batch_scene_pixels(new_frames)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(data['control']['p']['parent'], compile=False)
    old_parent = replay(old_frames, softmax(parent(data['control']['x'][-385:], training=False).numpy()))
    new_parent = replay(new_frames, softmax(parent(new_pixels, training=False).numpy()))
    for predicted, expected in ((old_parent, old_frames), (new_parent, new_frames)):
        for a, b in zip(predicted['frames'], expected, strict=True):
            check_replayed_parent(a, b)
    parent_batch = softmax(parent(data['control']['batch_pixels'], training=False).numpy())
    scenes = dict(parent=dict(original=old_parent, window1=new_parent))
    results = {}
    for arm, d in data.items():
        before = parent(d['x'], training=False).numpy()
        probe_before = parent(d['probes'], training=False).numpy()
        for seed in d['protocol']['seeds']:
            root = Path(f'runs/balloon-hue-{arm}-seed{seed}-20261011')
            model, after, probes, crop = verify_run(root, d, arm, seed, parent, before, probe_before)
            old = replay(old_frames, softmax(after[-385:]))
            new = replay(new_frames, softmax(model(new_pixels, training=False).numpy()))
            batch_logits = model(d['batch_pixels'], training=False).numpy()
            batch_probs = softmax(batch_logits)
            batch_correct = correct(batch_logits, d['batch_labels']).tolist()
            effect = difference(old, old_parent)
            negatives = [false_acceptances(v) for v in (probe_before, probes)]
            new_effect = difference(new, new_parent)
            gate = training_gate(effect, new_effect, negatives)
            name = f'{arm}{seed}'
            scenes[name] = dict(original=old, window1=new)
            results[name] = dict(run=str(root), artifacts_sha256={f:sha256(root/f) for f in
                ('student.keras', 'predictions.npz', 'report.json', 'provenance.json', 'epochs.csv')},
                original_metrics=checked_metrics(old), window1_metrics=checked_metrics(new),
                effect=effect, background_false_acceptances=negatives, old_crop_diagnostic=crop,
                batch_correct=batch_correct, batch_probabilities=batch_probs.tolist(),
                hard_blue_probabilities=batch_probs[-2:, 2].tolist(), new_effect=new_effect, training_scene_gate_pass=bool(gate))
            print(dict(name=name, gate=bool(gate), hard_blue=batch_probs[-2:, 2].tolist(),
                window1=new['metrics'], original=old['metrics'], lost=len(effect['lost_targets']),
                negatives=negatives), flush=True)
    eligible = all(results[f'hue{s}']['training_scene_gate_pass'] for s in (47, 48))
    for seed in (47, 48):
        results[f'hue{seed}']['against_matched_control'] = {
            key:difference(scenes[f'hue{seed}'][key], scenes[f'control{seed}'][key])
            for key in ('original', 'window1')}
    write_json(RAW, scenes)
    write_json(OUTPUT, dict(results=results, changed_rows=changed, protocol_sha256=sha256(PROTOCOL),
        raw=dict(path=str(RAW), sha256=sha256(RAW)), parent_batch_probabilities=parent_batch.tolist(),
        inputs_sha256={str(p):sha256(p) for p in (PROTOCOL, __file__, 'scripts/balloon_hue_trial_data.py',
            'scripts/train_balloon_hue.py', 'scripts/diagnose_clutter_round3_feature.py',
            'scripts/diagnose_clutter_stage_attribution.py', 'scripts/summarize_public_balloon_trial.py',
            'scripts/audit_balloon_context_distillation.py', TRAIN)},
        eligible_for_export_and_development=eligible,
        decision='Eligible for separate export parity and full-search repeatedDEV.' if eligible else
                 'Reject: paired TRAIN nonregression/recovery gate failed.',
        scope='Recomputed Keras scores on unchanged classifier-independent TRAIN geometry; fresh confidence selection and one-to-one IoU0.5 recount. Original64 plus6new public TRAIN photos; not independent accuracy. No full-search export or Pi qualification yet.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    audit()
