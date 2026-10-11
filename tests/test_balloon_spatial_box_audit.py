import numpy as np
import pytest

from scripts.audit_balloon_spatial_box import grouped_predictions
from scripts.train_balloon_quality import fold_ids


def examples():
    return [dict(source_image=str(i),source_group=str(i),source_sha256=str(i),
                 normalized_target=[.1,.2,.8,.9]) for i in range(30)]


def test_grouped_prediction_covers_every_row_without_held_label_in_fit():
    rows=examples()
    features=np.random.default_rng(7).normal(size=(30,4))
    before,folds=grouped_predictions(features,rows,dict(ridge_penalty=10.))
    assert before.shape==(30,4) and sum(r['hold_count'] for r in folds)==30
    selected=fold_ids(rows)==0
    changed=[dict(r,normalized_target=[.2,.3,.7,.8]) if yes else r for r,yes in zip(rows,selected)]
    after,_=grouped_predictions(features,changed,dict(ridge_penalty=10.))
    np.testing.assert_array_equal(before[selected],after[selected])


def test_cross_fold_source_hash_reuse_rejected():
    rows=examples()
    folds=fold_ids(rows)
    index=next(i for i,f in enumerate(folds) if f!=folds[0])
    rows[index]['source_sha256']=rows[0]['source_sha256']
    with pytest.raises(ValueError,match='leakage'):
        grouped_predictions(np.ones((30,4)),rows,dict(ridge_penalty=10.))


def test_single_source_fold_rejected():
    rows=[examples()[0]]*5
    with pytest.raises(ValueError,match='distinct'):
        grouped_predictions(np.ones((5,4)),rows,dict(ridge_penalty=10.))
