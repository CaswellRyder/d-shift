"""Reconstruct every paired training input and independently re-run saved checkpoints."""
import csv
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_scene_constraints import decisions
from scripts.clutter_round3_feature_data import PROTOCOL, tensors, hashes
from scripts.train_batch3_retraining import calculate

OUTPUT = Path('research/balloon-red-blue-20261008/clutter-round3-feature-20261011.json')


def recompute(before, after, labels, probe_after, probe_labels, scene_mask):
    if (before.shape != after.shape or before.shape != (2511, 3)
            or labels.shape != (2511,) or scene_mask.shape != (2511,)
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or not np.array_equal(np.flatnonzero(scene_mask), np.arange(2126, 2511))):
        raise ValueError('Invalid prediction shapes or scene boundary')
    result = calculate(before[:2074], after[:2074], labels[:2074], probe_after, probe_labels)
    result['scene_changed_rows'] = np.flatnonzero(
        decisions(before[scene_mask]) != decisions(after[scene_mask])).tolist()
    result['training_gate_pass'] &= not bool(result['scene_changed_rows'])
    return result


def eligible(results, seeds):
    if set(results) != {f'{a}{s}' for a in ('control', 'admit') for s in seeds}:
        raise ValueError('Incomplete matched experiment')
    return all(results[f'admit{s}']['training_gate_pass'] for s in seeds)


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'], compile=False)
    results = {}
    for arm in p['arms']:
        data = tensors(arm)
        before = parent(data['x'], training=False).numpy()
        probe_before = parent(data['probes'], training=False).numpy()
        for seed in p['seeds']:
            root = Path(f'runs/clutter-round3-feature-{arm}-seed{seed}-20261011')
            prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
            if (read_json(root/'status.json')['state'] != 'complete' or prov['settings'] != p
                    or prov['arm'] != arm or prov['seed'] != seed
                    or prov['tensor_sha256'] != hashes(data)
                    or prov['sample_identity'] != data['sample_identity']
                    or prov['scene_identities'] != data['scene_ids']
                    or prov['fixed_final_epoch'] is not True or prov['fresh_optimizer'] is not True
                    or any(prov[k] is not False for k in ('development_evaluated', 'test_evaluated', 'deployment_approved'))
                    or report['optimizer_steps'] != p['optimizer_steps']
                    or report['provenance_sha256'] != sha256(root/'provenance.json')
                    or report['predictions_sha256'] != sha256(root/'predictions.npz')
                    or report['model_sha256'] != sha256(root/'student.keras')
                    or any(sha256(path) != h for path, h in prov['sources_sha256'].items())):
                raise ValueError('Training identity or scope changed')
            required = dict(data['sources_sha256'])
            train_script = str(Path('scripts/train_clutter_round3_feature.py').resolve())
            # Python __file__ may be relative or absolute; require its resolved identity.
            resolved = {str(Path(k).resolve()): v for k, v in prov['sources_sha256'].items()}
            if (any(prov['sources_sha256'].get(k) != h for k, h in required.items())
                    or resolved.get(train_script) != sha256(train_script)):
                raise ValueError('Missing producer or input identity')
            with (root/'epochs.csv').open() as handle:
                trace = list(csv.DictReader(handle))
            if (len(trace) != p['epochs'] or [int(r['epoch']) for r in trace] != list(range(p['epochs']))
                    or not all(np.isfinite(float(r['loss'])) for r in trace)):
                raise ValueError('Incomplete fixed-epoch training trace')
            model = keras.models.load_model(root/'student.keras', compile=False)
            if model.count_params() != p['parameters'] or model.get_config() != parent.get_config():
                raise ValueError('Inference architecture changed')
            after = model(data['x'], training=False).numpy()
            probe_after = model(data['probes'], training=False).numpy()
            with np.load(root/'predictions.npz', allow_pickle=False) as stored:
                for k, value in dict(before=before, after=after, probe_before=probe_before, probe_after=probe_after).items():
                    if (not np.isfinite(stored[k]).all()
                            or not np.allclose(stored[k], value, atol=1e-5, rtol=0)):
                        raise ValueError('Checkpoint predictions do not reproduce')
                if (not np.array_equal(stored['labels'], data['y'])
                        or not np.array_equal(stored['probe_labels'], data['probe_labels'])):
                    raise ValueError('Stored labels changed')
            measured = recompute(before, after, data['y'], probe_after, data['probe_labels'], data['scene_mask'])
            if any(report[k] != value for k, value in measured.items()):
                raise ValueError('TRAIN gate does not reproduce')
            changed = [layer.name for layer in model.layers if any(
                not np.array_equal(a, b) for a, b in zip(layer.get_weights(),
                    parent.get_layer(layer.name).get_weights(), strict=True))]
            if not any(name != 'classifier' for name in changed):
                raise ValueError('No feature layer learned')
            results[f'{arm}{seed}'] = dict(run=str(root), report_sha256=sha256(root/'report.json'),
                epochs_sha256=sha256(root/'epochs.csv'), **report, changed_layers=changed,
                lost_old_identities=[{k: data['rows'][i][k] for k in ('source_image', 'sha256', 'label', 'view')}
                                     for i in measured['lost_old_rows']])
            print(dict(audited=f'{arm}{seed}', gate=measured['training_gate_pass'],
                new_correct=sum(measured['probe_correct']), lost=len(measured['lost_old_rows']),
                scene_changes=len(measured['scene_changed_rows'])), flush=True)
    passed = eligible(results, p['seeds'])
    write_json(OUTPUT, dict(results=results, protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        data_helper_sha256=sha256('scripts/clutter_round3_feature_data.py'),
        eligible_for_full_scene_comparison=passed,
        decision='Proceed to export parity and separate TRAIN full-scene comparison.' if passed else
                 'Reject for promotion: paired feature retraining failed the preregistered TRAIN retention gate.',
        scope='Independently reproduced four fixed80epoch checkpoints,12560steps each. Correlated TRAIN crops and scene proposals only; not independent accuracy or Pi speed.',
        development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))
    print(dict(eligible_for_full_scene_comparison=passed), flush=True)


if __name__ == '__main__':
    audit()
