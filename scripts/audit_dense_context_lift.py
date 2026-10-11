"""Verify warm-start equivalence on reviewed TRAIN crops before any capacity trial."""
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.runtime import softmax
from scripts.blue_photo_expansion_data import inputs
from scripts.balloon_dense_context_lift import build, cost

OUTPUT = Path('research/balloon-red-blue-20261008/dense-context-lift-initialization-20261011.json')
PROTOCOL = Path('configs/blue-photo-convergence-20261011.json')


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    p = read_json(PROTOCOL)
    if sha256(p['parent']) != p['parent_sha256']:
        raise ValueError('Frozen parent changed')
    _, _, _, tensors = inputs('admit')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    parent = keras.models.load_model(p['parent'], compile=False)
    previous = [a.copy() for a in parent.get_weights()]
    model = build(parent)
    if any(not np.array_equal(a, b) for a, b in zip(previous, parent.get_weights(), strict=True)):
        raise ValueError('Parent was modified')
    checks = []
    for name, x in zip(('train', 'old_blue', 'old_clutter', 'real_clutter', 'new_clutter', 'blue_views', 'new_photos'), tensors, strict=True):
        a, b = parent(x, training=False).numpy(), model(x, training=False).numpy()
        sa, sb = softmax(a), softmax(b)
        delta = float(np.max(np.abs(a-b)))
        same_class = np.array_equal(a.argmax(1), b.argmax(1))
        same_accept = np.array_equal(sa.max(1) >= .8, sb.max(1) >= .8)
        if not np.isfinite(b).all() or delta > .0001 or not same_class or not same_accept:
            raise ValueError('Initial function changed beyond numerical tolerance')
        checks.append(dict(group=name, rows=len(x), max_logit_delta=delta,
            max_probability_delta=float(np.max(np.abs(sa-sb))), same_class=same_class, same_acceptance=same_accept,
            tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    write_json(OUTPUT, dict(parent=dict(path=p['parent'], sha256=p['parent_sha256']),
        checks=checks, total_rows=sum(len(x) for x in tensors), cost=cost(),
        source_sha256={path: sha256(path) for path in ('scripts/balloon_dense_context_lift.py', __file__,
            'scripts/blue_photo_expansion_data.py', 'scripts/blue_balloon_candidate_data.py', 'src/dtr/models.py')},
        protocol_sha256=sha256(PROTOCOL), threshold=.8, allowed_max_logit_delta=.0001,
        next_action='Matched compact-versus-dense training on the same reviewed curriculum with fixed schedules, followed by scene accuracy and separate Pi latency checks.',
        scope='Numerical initial-function equivalence only. Extra capacity is a hypothesis, not demonstrated accuracy or speed improvement.',
        trained=False, test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(rows=sum(len(x) for x in tensors), max_delta=max(r['max_logit_delta'] for r in checks), cost=cost()))


if __name__ == '__main__':
    audit()
