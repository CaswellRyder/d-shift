import numpy as np
import pytest

from scripts.balloon_hog_gate import features,fit,fold_fit,score,source_folds


def test_hog_shape_and_input_contract():
    image = np.zeros((64,64,3),np.uint8)
    actual = features([image,image.astype(np.float32)])
    assert actual.shape == (2,324) and np.array_equal(actual[0],actual[1])
    with pytest.raises(ValueError,match='quantize'):
        features([image+.5])
    with pytest.raises(ValueError,match='canonical'):
        features([np.zeros((32,32,3),np.uint8)])


def test_source_aliases_pixel_duplicates_and_transitive_links_stay_together():
    rows = [dict(source_image='train/frame_1_jpg.rf.abc.jpg',sha256='a'),
        dict(source_image='train/frame_1_jpg.rf.def.jpg',sha256='b'),
        dict(source_image='another.png',sha256='b'),dict(source_image='other.png',sha256='c')]
    groups,folds = source_folds(rows)
    assert groups[0] == groups[1] == groups[2] and groups[3] != groups[0]
    assert folds[0] == folds[1] == folds[2]
    assert source_folds(list(reversed(rows)))[0] == list(reversed(groups))


def test_fit_threshold_keeps_all_fitting_foregrounds():
    x = np.array([[0.,0.],[1.,0.],[10.,1.],[11.,1.]])
    labels = np.array([0,0,1,2])
    model = fit(x,labels)
    assert np.all(score(x,model)[labels>0] >= model['threshold'])


def test_withheld_values_cannot_change_fitted_model():
    x = np.array([[0.,0.],[1.,1.],[10.,10.],[20.,20.]])
    labels,folds = np.array([0,1,0,2]),np.array([0,0,1,1])
    a,_,keep = fold_fit(x,labels,folds,1)
    changed = x.copy()
    changed[~keep] = 1e10
    b,_,_ = fold_fit(changed,labels,folds,1)
    assert all(np.array_equal(a[k],b[k]) for k in a)
