"""Add reviewed TRAIN-only background masks; preserve sparse data and holdout."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from dtr.data import read_json, sha256, write_json
from scripts.balloon_dense_complete import encode_complete, reviewed_indices
from scripts.train_balloon_dense_proposals import checked_data


def prepare(root, review_path, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    base,_,targets,folds,groups = checked_data(root)
    if 'complete_background' in base:
        raise ValueError('Require original sparse dataset')
    review = read_json(review_path)
    approved = reviewed_indices(base,folds,groups,review['queue'],review_path,
                                sha256(root/'manifest.json'))
    updated = targets.copy()
    for i in approved:
        row = base['frames'][i]
        updated[i],_ = encode_complete(row['truth'],row['negative_regions'])
    if not np.array_equal(updated[folds==0],targets[folds==0]):
        raise ValueError('Holdout supervision changed')
    output.mkdir(parents=True)
    rows = []
    for i,row in enumerate(base['frames']):
        shutil.copyfile(root/row['path'],output/row['path'])
        rows.append(dict(row,supervised_heat_cells=int((updated[i,:,:,2:4]>0).sum())))
    np.save(output/'targets.npy',updated,allow_pickle=False)
    doc = dict(base,frames=rows,targets_sha256=sha256(output/'targets.npy'),
        script_sha256=sha256(__file__),
        scope='Sparse TRAIN plus visually reviewed complete-fit-frame background outside 16px guards. Holdout unchanged; no independent qualification.',
        complete_background=dict(source_root=str(root),source_manifest_sha256=sha256(root/'manifest.json'),
            review=str(review_path),review_sha256=sha256(review_path),queue=review['queue'],
            queue_sha256=sha256(review['queue']),guard_pixels=16,
            encoding_sha256=sha256('scripts/balloon_dense_complete.py')))
    write_json(output/'manifest.json',doc)
    checked_data(output)
    return dict(complete_fit_frames=len(approved),unchanged_holdout_frames=int((folds==0).sum()),
        extra_heat_cells=int((updated[:,:,:,2:4]-targets[:,:,:,2:4]).sum()),
        manifest_sha256=sha256(output/'manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','review','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    print(prepare(a.data,a.review,a.output))
