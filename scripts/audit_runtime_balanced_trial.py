"""Independently rebuild TRAIN tensors, predictions, gates and paired identities."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.train_runtime_balanced_trial import PROTOCOL, GROUPS, inputs, run_path, calculate

OUTPUT = Path('research/balloon-red-blue-20261008/runtime-balanced-training-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Parent changed')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'],compile=False)
    results,common,probe_hashes,matched_counts = {},None,None,None
    for mode in p['arms']:
        rows,labels,ordinary,tensors,runtime_labels = inputs(mode)
        identity = [{k:r[k] for k in ('source_image','sha256','label','preserve','view')} for r in rows]
        hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in tensors]
        counts = np.bincount(labels,minlength=3).tolist()
        if (common is not None and (common != identity[:1946] or probe_hashes != hashes[1:]
                or counts != matched_counts)):
            raise ValueError('Matched base, class counts or probes differ')
        if len(rows) != 2074 or ordinary.sum() != 1602:
            raise ValueError('Dose changed')
        common,probe_hashes,matched_counts = identity[:1946],hashes[1:],counts
        teacher = [parent(x,training=False).numpy() for x in tensors]
        for seed in p['seeds']:
            run = run_path(mode,seed)
            prov,report = read_json(run/'provenance.json'),read_json(run/'report.json')
            if (read_json(run/'status.json')['state'] != 'complete' or prov['mode'] != mode or prov['seed'] != seed
                    or prov['settings'] != p or prov['sample_identity'] != identity or prov['tensor_sha256'] != hashes
                    or prov['parameters'] != 7763 or report['parameters'] != 7763
                    or prov['optimizer_reset'] is not True or prov['final_epoch_only'] is not True
                    or report['optimizer_steps'] != 52000
                    or any(sha256(k) != h for k,h in prov['sources_sha256'].items())):
                raise ValueError('Training recipe or inputs changed')
            for key,name in (('model_sha256','student.keras'),('predictions_sha256','predictions.npz'),
                    ('provenance_sha256','provenance.json')):
                if report[key] != sha256(run/name):
                    raise ValueError('Output identity changed')
            model = keras.models.load_model(run/'student.keras',compile=False)
            if model.count_params() != 7763:
                raise ValueError('Architecture changed')
            with np.load(run/'predictions.npz') as saved:
                arrays = {k:saved[k] for k in saved.files}
            for name,expected in dict(labels=labels,ordinary=ordinary,indoor_labels=labels[1786:1854],runtime_labels=runtime_labels).items():
                if not np.array_equal(arrays[name],expected):
                    raise ValueError('Labels/protection changed')
            deltas = []
            for name,x,before in zip(GROUPS,tensors,teacher,strict=True):
                after = model(x,training=False).numpy()
                if (not np.allclose(before,arrays[name+'before'],rtol=0,atol=1e-5)
                        or not np.allclose(after,arrays[name+'after'],rtol=0,atol=1e-5)):
                    raise ValueError('Checkpoint or teacher does not reproduce')
                deltas.append(float(np.max(np.abs(after-arrays[name+'after']))))
            with zipfile.ZipFile(run/'student.keras') as archive:
                optimizer = json.loads(archive.read('config.json'))['compile_config']['optimizer']
            schedule = optimizer['config']['learning_rate']
            if (optimizer['class_name'] != 'Adam' or schedule['class_name'] != 'CosineDecay'
                    or not np.isclose(schedule['config']['initial_learning_rate'],p['learning_rate'],rtol=1e-6,atol=0)
                    or schedule['config']['decay_steps'] != p['decay_steps'] or schedule['config']['alpha'] != p['alpha']
                    or schedule['config'].get('warmup_target') is not None
                    or schedule['config'].get('warmup_steps',0) != 0):
                raise ValueError('Saved optimizer differs')
            with (run/'epochs.csv').open() as stream:
                history = list(csv.DictReader(stream))
            losses = np.array([float(row['loss']) for row in history])
            if [int(row['epoch']) for row in history] != list(range(p['epochs'])) or not np.isfinite(losses).all():
                raise ValueError('Incomplete history')
            fresh = calculate(arrays)
            if any(report[k] != v for k,v in fresh.items()):
                raise ValueError('Training gates do not recount')
            results[f'{mode}{seed}'] = dict(path=str(run),**report,saved_optimizer=optimizer,
                history_sha256=sha256(run/'epochs.csv'),final_loss=float(losses[-1]),
                last40_max_loss=float(losses[-40:].max()),last40_median_loss=float(np.median(losses[-40:])),
                max_checkpoint_delta=max(deltas))
    eligible = all(results[f'admit{s}']['training_gate_pass'] for s in p['seeds'])
    write_json(OUTPUT,dict(trials=results,eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL),script_sha256=sha256(__file__),
        decision='Export for frozen scene evaluation' if eligible else 'No export: paired training gate failed',
        scope='TRAIN reproduction only; matched classes/dose, not independent accuracy.',
        test_evaluated=False,pi_tested=False,deployment_approved=False,qualification_achieved=False))
    print(dict(eligible=eligible,trials={k:dict(gate=r['training_gate_pass'],
        objects=r['object_background_correct'],runtime_original=r['runtime_crop_original_passed'],
        runtime_mirror=r['runtime_crop_mirror_passed'],indoor=r['indoor_passed']) for k,r in results.items()}))


if __name__ == '__main__':
    audit()
