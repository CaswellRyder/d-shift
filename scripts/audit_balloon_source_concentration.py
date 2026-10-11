"""Measure source repetition in guarded TRAIN inputs, not independent sample size."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.fit_blue_prototype_rejection import checked_inputs
from scripts.balloon_scene_constraints import scene_pixels, RAW

OUTPUT = Path('research/balloon-red-blue-20261008/training-source-concentration-20261011.json')


def summarize(rows, pixels):
    if len(rows) != len(pixels) or not rows or not np.isfinite(pixels).all():
        raise ValueError('Require complete finite training tensors')
    if any(r['split'] != 'train' or 'IMG_' in r['source_image'] for r in rows):
        raise ValueError('Evaluation-only source in training inventory')
    sources = Counter(r['source_image'] for r in rows)
    digests = [hashlib.sha256(x.tobytes()).hexdigest() for x in pixels]
    pixel_labels = defaultdict(set)
    for digest, row in zip(digests, rows, strict=True):
        pixel_labels[digest].add(row['label'])
    per_class = {}
    for label in sorted({r['label'] for r in rows}):
        selected = [i for i, r in enumerate(rows) if r['label'] == label]
        counts = Counter(rows[i]['source_image'] for i in selected)
        per_class[label] = dict(rows=len(selected), named_sources=len(counts),
            unique_tensors=len({digests[i] for i in selected}),
            largest_source_rows=max(counts.values()),
            concentration_equivalent_sources=len(selected)**2/sum(v*v for v in counts.values()))
    return dict(rows=len(rows), named_sources=len(sources), unique_tensors=len(set(digests)),
        repeated_tensor_rows=len(rows)-len(set(digests)), per_class=per_class,
        source_counts=dict(sorted(sources.items())), views=dict(Counter(r['view'] for r in rows)),
        source_prefix_rows=dict(Counter(r['source_image'].split('/')[0] for r in rows)),
        rows_with_original_source_hash=sum(bool(r.get('source_sha256')) for r in rows),
        known_original_source_hashes=len({r['source_sha256'] for r in rows if r.get('source_sha256')}),
        exact_tensor_label_conflicts={k:sorted(v) for k,v in pixel_labels.items() if len(v)>1},
        concentration_equivalent_sources=len(rows)**2/sum(v*v for v in sources.values()))


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows, labels, pixels, _, _, previous = checked_inputs()
    if len(rows) != 2074 or labels.shape != (2074,):
        raise ValueError('Frozen curriculum changed')
    result = summarize(rows, pixels)
    _, scenes = scene_pixels()
    old_sources = {r['source_image'] for r in rows}
    scene_sources = {r['source'] for r in scenes}
    write_json(OUTPUT, dict(curriculum=result,
        scene_proposals=dict(rows=len(scenes), named_sources=len(scene_sources),
            existing_curriculum_sources=len(old_sources & scene_sources),
            named_source_union=len(old_sources | scene_sources)),
        provenance=dict(path=str(previous), sha256=sha256(previous)),
        original_provenance_sources=read_json(previous)['sources_sha256'],
        frozen_scene_sha256=sha256(RAW), script_sha256=sha256(__file__),
        scope='TRAIN-only inventory. Filenames and source hashes are not independent recording sessions. Concentration-equivalent count is N^2/sum(source_count^2), a descriptive repetition measure, NOT statistical effective sample size. Synthetic views and related video frames do not establish scene diversity.',
        weights_changed=False, training_performed=False, development_evaluated=False,
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print({k:v for k,v in result.items() if k not in ('source_counts', 'exact_tensor_label_conflicts')}, flush=True)
    print(dict(exact_tensor_label_conflicts=len(result['exact_tensor_label_conflicts'])), flush=True)


if __name__ == '__main__':
    audit()
