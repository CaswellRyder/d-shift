"""Reproduce fixed paired TRAIN checkpoints and admission gates, not qualify."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.audit_blue_balloon_batch4 import checked_extra
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.grid_clutter_data import load_crop
from scripts.train_batch3_retraining import LABELS, choose_controls, calculate
from scripts.train_batch4_retraining import PROTOCOL

OUTPUT = Path('research/balloon-red-blue-20261008/batch4-retraining-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
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
        selected = sorted(extra if arm == 'admit' else choose_controls(rows, extra), key=lambda r: LABELS.index(r['label']))
        added = [dict(r, preserve=False, view='original') for _ in range(4) for r in selected]
        x = np.concatenate([old_pixels, np.stack([load_crop(r)[0] for r in added])])
        y = np.concatenate([labels, [LABELS.index(r['label']) for r in added]])
        ordinary = np.array([r['preserve'] for r in rows]+[False]*len(added), bool)
        hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in (x, y, ordinary, probes, probe_labels)]
        for seed in p['seeds']:
            root = Path(f'runs/batch4-retraining-{arm}-seed{seed}-20261011')
            prov, report = read_json(root/'provenance.json'), read_json(root/'report.json')
            if (read_json(root/'status.json')['state'] != 'complete' or prov['settings'] != p
                    or prov['arm'] != arm or prov['seed'] != seed or prov['tensor_sha256'] != hashes
                    or prov['sample_identity'] != [{k: r[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for r in rows+added]
                    or report['provenance_sha256'] != sha256(root/'provenance.json')
                    or report['predictions_sha256'] != sha256(root/'predictions.npz')
                    or report['model_sha256'] != sha256(root/'student.keras')
                    or report['optimizer_steps'] != 53600
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
            lost = [{k: rows[i][k] for k in ('source_image', 'sha256', 'label', 'view')} for i in measured['lost_old_rows']]
            results[f'{arm}{seed}'] = dict(run=str(root), report_sha256=sha256(root/'report.json'), **report,
                lost_old_identities=lost, probes=[dict(id=r['id'], label=r['label'], scores=s.tolist())
                    for r, s in zip(extra, softmax(probe_after), strict=True)])
            print(dict(audited=f'{arm}{seed}', gate=report['training_gate_pass'], new_pass=sum(report['probe_correct'])), flush=True)
    eligible = all(results[f'admit{s}']['training_gate_pass'] for s in p['seeds'])
    result = dict(protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__), results=results,
        eligible_for_full_scene_comparison=eligible,
        scope='Four fixed400epoch checkpoints, each53600steps; matched real-photo additions versus old-class controls. All TRAIN inputs and Keras outputs reproduced. Counts are correlated crop views, not full-scene or independent accuracy.',
        development_evaluated=False, test_evaluated=False, pi_tested=False, deployment_approved=False,
        qualification_achieved=False)
    write_json(OUTPUT, result)
    print(dict(eligible_for_full_scene_comparison=eligible), flush=True)


if __name__ == '__main__':
    audit()
