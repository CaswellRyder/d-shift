"""Verify matched TRAIN inputs and independently recompute blue-expansion metrics."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.openimages_blue_expansion import data
from scripts.grid_clutter_data import load_crop
from scripts.train_balloon_blue_expansion import PROTOCOL, training_gate
from scripts.train_balloon_reviewed_clutter import CLASSES
from scripts.balloon_positive_protection import retention


def counts(blue):
    if blue.shape != (23, 3) or not np.isfinite(blue).all():
        raise ValueError('Invalid blue probe array')
    return int((blue.argmax(1) == 2).sum()), int(((blue.argmax(1) == 2) & (softmax(blue)[:, 2] >= .8)).sum())


def checked_run(mode, seed):
    protocol = read_json(PROTOCOL)
    run = Path(f'runs/balloon-blue-expansion2-{mode}-seed{seed}-20261011')
    p, r = read_json(run / 'provenance.json'), read_json(run / 'report.json')
    if (read_json(run / 'status.json')['state'] != 'complete' or p['mode'] != mode or p['seed'] != seed
            or p['settings'] != protocol or p['parameters'] != 7763
            or r['model_sha256'] != sha256(run / 'student.keras')
            or r['predictions_sha256'] != sha256(run / 'predictions.npz')
            or r['provenance_sha256'] != sha256(run / 'provenance.json')
            or any(sha256(path) != h for path, h in p['sources_sha256'].items())
            or any(sha256(protocol[name]) != protocol[name + '_sha256'] for name in ('parent', 'teacher'))):
        raise ValueError('Trial identities changed')
    rows, extra, clutter = data(mode)
    identity = [{k: row[k] for k in ('source_image', 'sha256', 'label', 'preserve', 'view')} for row in rows]
    x = np.stack([load_crop(row)[0] for row in rows])
    blue = np.stack([load_crop(row)[0] for row in extra])
    probes = np.stack([load_crop(dict(row, view=v))[0] for v in ('original', 'blue_surface') for row in clutter])
    if (identity != p['sample_identity'] or any(hashlib.sha256(t.tobytes()).hexdigest() != p[k]
            for k, t in (('input_tensor_sha256', x), ('blue_tensor_sha256', blue), ('probe_tensor_sha256', probes)))):
        raise ValueError('Training tensors changed')
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        a = {k: saved[k] for k in saved.files}
    labels = np.array([CLASSES.index(row['label']) for row in rows])
    ordinary = np.array([row['preserve'] for row in rows], bool)
    correct, accepted = counts(a['blue_after'])
    if (not np.array_equal(a['labels'], labels) or not np.array_equal(a['ordinary'], ordinary)
            or a['probe_after'].shape != (22, 3) or not np.isfinite(a['probe_after']).all()
            or r['retention'] != retention(a['before'], a['after'], labels, ordinary)
            or r['warm_start_retention'] != retention(a['start'], a['after'], labels, ordinary)
            or r['reviewed_background_correct'] != int((a['probe_after'].argmax(1) == 0).sum())
            or r['new_blue_correct'] != correct or r['new_blue_accepted'] != accepted
            or r['initial_blue_accepted'] != counts(a['blue_before'])[1]
            or r['training_gate_pass'] != training_gate(r, protocol)):
        raise ValueError('Trial metrics do not reproduce')
    return run, p, r, a, (x, blue, probes)


def audit(output):
    if Path(output).exists():
        raise FileExistsError(output)
    trials = {}
    for mode in ('control', 'admit'):
        for seed in (47, 48):
            run, p, r, _, _ = checked_run(mode, seed)
            trials[f'{mode}{seed}'] = dict(r, path=str(run), input_tensor_sha256=p['input_tensor_sha256'])
        if trials[f'{mode}47']['input_tensor_sha256'] != trials[f'{mode}48']['input_tensor_sha256']:
            raise ValueError('Seeds trained on different rows')
    eligible = all(trials[f'admit{s}']['training_gate_pass'] for s in (47, 48))
    write_json(output, dict(trials=trials, eligible_for_development=eligible,
        protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        decision='Fixed24 TRAIN gate only. Failed candidates are not exported or promoted.',
        test_evaluated=False, development_evaluated=False, deployment_approved=False,
        pi_tested=False, qualification_achieved=False))
    print(dict(eligible_for_development=eligible,
        trials={k: (v['new_blue_accepted'], v['reviewed_background_correct']) for k, v in trials.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
