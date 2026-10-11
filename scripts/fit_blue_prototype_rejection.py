"""Fit one fixed negative prototype bank while freezing the entire classifier."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_object_round2_data import checked_extra
from scripts.blue_prototype_rejection import adjust, correction, fit
from scripts.grid_clutter_data import load_crop
from scripts.runtime_balanced_data import inputs

PROTOCOL = Path('configs/blue-prototype-rejection-20261011.json')
RUN = Path('runs/blue-prototype-rejection-20261011')
OUTPUT = Path('research/balloon-red-blue-20261008/blue-prototype-training-20261011.json')


def feature_model():
    import keras
    protocol = read_json(PROTOCOL)
    if sha256(protocol['parent']) != protocol['parent_sha256']:
        raise ValueError('Frozen parent identity changed')
    parent = keras.models.load_model(protocol['parent'], compile=False)
    if parent.count_params() != 7763 or parent.get_layer('spatial_features').output.shape[-1] != 128:
        raise ValueError('Architecture changed')
    return keras.Model(parent.input, [parent.get_layer('spatial_features').output, parent.output])


def checked_inputs():
    previous = Path('runs/runtime-balanced-trial-admit-seed47-20261011/provenance.json')
    receipt = read_json(previous)
    rows, labels, _, tensors, _ = inputs('admit')
    if (len(rows) != 2074 or len(tensors) != 12
            or any(r['split'] != 'train' or 'IMG_' in r['source_image'] for r in rows)
            or [{k:r[k] for k in ('source_image','sha256','label','preserve','view')} for r in rows] != receipt['sample_identity']
            or [hashlib.sha256(x.tobytes()).hexdigest() for x in tensors] != receipt['tensor_sha256']
            or any(sha256(p) != h for p,h in receipt['sources_sha256'].items())):
        raise ValueError('Guarded TRAIN curriculum changed')
    extra = checked_extra()
    if len(extra) != 12 or any(r['label'] != 'background' for r in extra):
        raise ValueError('Prototype review changed')
    return rows, labels, tensors[0], extra, np.stack([load_crop(r)[0] for r in extra]), previous


def run():
    if RUN.exists() or OUTPUT.exists():
        raise FileExistsError(RUN)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    rows, labels, pixels, extra, centers, previous = checked_inputs()
    model = feature_model()
    f, z = [v.numpy() for v in model(pixels, training=False)]
    c, cz = [v.numpy() for v in model(centers, training=False)]
    bank = fit(f, labels, c, cz)
    delta, center_delta = correction(f, bank), correction(c, bank)
    old, new = softmax(z), adjust(softmax(z), delta)
    center_before, center_after = softmax(cz), adjust(softmax(cz), center_delta)
    safe = bool(np.all(delta[labels > 0] == 0) and np.all(center_after.argmax(axis=1) == 0)
        and np.array_equal(new[:,1], old[:,1]) and np.all(new[:,2] <= old[:,2]))
    RUN.mkdir(parents=True)
    np.savez_compressed(RUN/'bank.npz', **bank)
    np.savez_compressed(RUN/'training.npz', features=f, logits=z, labels=labels,
        center_features=c, center_logits=cz, delta=delta, center_delta=center_delta, after=new, center_after=center_after)
    write_json(OUTPUT, dict(protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(__file__),
        parent_sha256=read_json(PROTOCOL)['parent_sha256'],
        previous_provenance=dict(path=str(previous),sha256=sha256(previous)),
        helpers_sha256={p:sha256(p) for p in ('scripts/blue_prototype_rejection.py', 'scripts/runtime_balanced_data.py',
            'scripts/blue_object_round2_data.py', 'scripts/grid_clutter_data.py')},
        bank=dict(path=str(RUN/'bank.npz'),sha256=sha256(RUN/'bank.npz')),
        training=dict(path=str(RUN/'training.npz'),sha256=sha256(RUN/'training.npz')),
        rows=len(rows), protected_positive_rows=int((labels > 0).sum()),
        changed_positive_rows=int((delta[labels > 0] > 0).sum()), prototypes=len(c),
        usable_prototypes=int((bank['radii2'] > 1e-12).sum()),
        prototype_results=[dict(source=r['source_image'],sha256=r['sha256'],before=a.tolist(),after=b.tolist(),
            radius_squared=float(radius),correction=float(value))
            for r,a,b,radius,value in zip(extra,center_before,center_after,bank['radii2'],center_delta,strict=True)],
        changed_training_rows=int((delta > 0).sum()), training_gate_pass=safe,
        scope='Deterministic TRAIN-only prototype fit; all classifier weights frozen. No development pixels/labels used in fitting, no neural runtime export or speed claim.',
        test_evaluated=False, deployment_approved=False, qualification_achieved=False))
    print(dict(training_gate_pass=safe, positive_rows=int((labels>0).sum()), changed_positive_rows=int((delta[labels>0]>0).sum()),
        backgrounds_passed=int((center_after.argmax(axis=1)==0).sum()), changed_training_rows=int((delta>0).sum())), flush=True)


if __name__ == '__main__':
    run()
