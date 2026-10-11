"""Verify fixed400epoch checkpoints and qualify only for TRAIN scene comparison."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_balloon_batch3_data import checked_extra
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.grid_clutter_data import load_crop
from scripts.train_batch3_retraining import LABELS, calculate
from scripts.train_background_convergence import PROTOCOL

OUTPUT = Path('research/balloon-red-blue-20261008/background-convergence-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    if sha256('research/balloon-red-blue-20261008/background-preservation-20261011.json') != p['prior_audit_sha256']:
        raise ValueError('Prior paired80epoch audit changed')
    rows, labels, old_pixels, _, _, _ = checked_inputs()
    extra = checked_extra()
    probes = np.stack([load_crop(r)[0] for r in extra])
    probe_labels = np.array([LABELS.index(r['label']) for r in extra])
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    parent = keras.models.load_model(p['parent'], compile=False)
    before, probe_before = parent(old_pixels, training=False).numpy(), parent(probes, training=False).numpy()
    results = {}
    for arm in p['arms']:
        selected = sorted(extra, key=lambda r: LABELS.index(r['label']))
        added = [dict(r, preserve=False, view='original') for _ in range(4) for r in selected]
        x = np.concatenate([old_pixels, np.stack([load_crop(r)[0] for r in added])])
        y = np.concatenate([labels, [LABELS.index(r['label']) for r in added]])
        ordinary = np.array([r['preserve'] for r in rows]+[False]*len(added), bool)
        hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in (x, y, ordinary, probes, probe_labels)]
        teacher = parent(x, training=False).numpy()
        target = np.column_stack([np.eye(3)[y], teacher, ordinary, np.arange(len(x)) < len(rows)]).astype(np.float32)
        for seed in p['seeds']:
            root = Path(f'runs/background-convergence-{arm}-seed{seed}-20261011')
            prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
            if (read_json(root/'status.json')['state'] != 'complete' or prov['settings'] != p
                    or prov['arm'] != arm or prov['seed'] != seed or prov['tensor_sha256'] != hashes
                    or prov['target_sha256'] != hashlib.sha256(target.tobytes()).hexdigest()
                    or prov['sample_identity'] != [{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows+added]
                    or report['provenance_sha256'] != sha256(root/'provenance.json')
                    or report['predictions_sha256'] != sha256(root/'predictions.npz')
                    or report['model_sha256'] != sha256(root/'student.keras')
                    or report['optimizer_steps'] != 52800
                    or any(sha256(path) != h for path, h in prov['sources_sha256'].items())):
                raise ValueError('Training, inputs or checkpoint identity changed')
            model = keras.models.load_model(root/'student.keras', compile=False)
            if model.count_params() != 7763:
                raise ValueError('Architecture changed')
            after, probe_after = model(old_pixels, training=False).numpy(), model(probes, training=False).numpy()
            with np.load(root/'predictions.npz', allow_pickle=False) as stored:
                for name, value in dict(before=before, after=after, probe_before=probe_before, probe_after=probe_after).items():
                    if not np.isfinite(stored[name]).all() or not np.allclose(stored[name], value, atol=1e-5, rtol=0):
                        raise ValueError('Saved predictions differ from checkpoint')
                if not np.array_equal(stored['labels'], labels) or not np.array_equal(stored['probe_labels'], probe_labels):
                    raise ValueError('Probe labels changed')
            measured = calculate(before, after, labels, probe_after, probe_labels)
            if any(report[k] != v for k, v in measured.items()):
                raise ValueError('Reported TRAIN gate does not reproduce')
            prior = Path(f'runs/batch3-retraining-admit-seed{seed}-20261011')
            previous = read_json(prior/'provenance.json')
            if previous['tensor_sha256'] != hashes or previous['sample_identity'] != prov['sample_identity']:
                raise ValueError('Unmatched historical control inputs')
            lost = [{k: rows[i][k] for k in ('source_image', 'sha256', 'label', 'view')} for i in measured['lost_old_rows']]
            results[f'{arm}{seed}'] = dict(run=str(root), report_sha256=sha256(root/'report.json'), **report,
                lost_old_identities=lost, historical_output_comparison_not_applicable=True, probes=[dict(id=r['id'], label=r['label'], scores=s.tolist())
                    for r, s in zip(extra, softmax(probe_after), strict=True)])
            print(dict(audited=f'{arm}{seed}', gate=report['training_gate_pass'], new_pass=sum(report['probe_correct'])), flush=True)
    eligible_arms = [a for a in p['arms'] if all(results[f'{a}{s}']['training_gate_pass'] for s in p['seeds'])]
    eligible = bool(eligible_arms)
    result = dict(protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__), results=results,
        eligible_for_full_scene_comparison=eligible, eligible_arms=eligible_arms,
        scope='Four fixed400epoch checkpoints, each52800steps; matched positive-only versus background-protected loss on identical admitted real-photo data. All TRAIN inputs and Keras outputs reproduced. Counts are correlated crop views, not full-scene or independent accuracy.',
        development_evaluated=False, test_evaluated=False, pi_tested=False, deployment_approved=False,
        qualification_achieved=False)
    write_json(OUTPUT, result)
    print(dict(eligible_for_full_scene_comparison=eligible), flush=True)


if __name__ == '__main__':
    audit()
