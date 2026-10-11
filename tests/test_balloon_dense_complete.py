import copy

import numpy as np
import pytest

from dtr.data import sha256, write_json
from scripts.balloon_dense_complete import encode_complete, reviewed_indices
from scripts.prepare_balloon_dense_proposals import encode


def test_complete_masks_add_only_background_outside_full_cell_guard():
    truth = [dict(label='blue_balloon',box=[64,64,96,96])]
    sparse,_ = encode(truth,[])
    complete,_ = encode_complete(truth,[])
    np.testing.assert_array_equal(complete[:,:,:2],sparse[:,:,:2])
    np.testing.assert_array_equal(complete[:,:,4:],sparse[:,:,4:])
    # Balloon ends at 96, guard at 112: cell 96..112 remains unknown.
    np.testing.assert_array_equal(complete[5,6,2:4],[0,0])
    np.testing.assert_array_equal(complete[5,7,2:4],[1,1])
    np.testing.assert_array_equal(complete[0,0,2:4],[1,1])
    assert np.all(complete[:,:,2:4]>=sparse[:,:,2:4])


def test_empty_complete_frame_is_explicit_background_but_sparse_empty_is_unknown():
    complete,_ = encode_complete([],[])
    sparse,_ = encode([],[])
    assert not sparse.any()
    assert np.all(complete[:,:,2:4]==1)
    assert not complete[:,:,:2].any() and not complete[:,:,4:].any()
    with pytest.raises(ValueError,match='guard'):
        encode_complete([],[],guard=0)


@pytest.fixture
def review_case(tmp_path):
    base = dict(frames=[dict(panel='engdes2-fixture',source=f'frame_{i}.jpg',
        truth=[dict(label='red_balloon',box=[64,64,96,96])]) for i in range(3)])
    folds, groups = [1,2,0], ['a','b','heldout']
    queue = dict(source_manifest_sha256='basehash',sheets={},frames=[
        dict(base['frames'][i],index=i,fold=folds[i],source_group=groups[i]) for i in (0,1)])
    qp,rp = tmp_path/'queue.json',tmp_path/'review.json'
    write_json(qp,queue)
    review = dict(queue=str(qp),queue_sha256=sha256(qp),source_manifest_sha256='basehash',
        guard_pixels=16,training_approved=True,test_evaluated=False,deployment_approved=False,
        complete_indices=[0],excluded={'1':'Unboxed partial; sparse only.'})
    write_json(rp,review)
    return base,folds,groups,queue,review,qp,rp


def check_case(case):
    base,folds,groups,_,_,qp,rp = case
    return reviewed_indices(base,folds,groups,qp,rp,'basehash')


def test_review_admits_only_explicit_complete_fit_frames(review_case):
    assert check_case(review_case)=={0}


@pytest.mark.parametrize('change', ['heldout','duplicate','omitted','overlap','not_approved','hash'])
def test_review_rejects_bad_decisions(review_case,change):
    *_,review,qp,rp = review_case
    if change=='heldout':
        review['complete_indices'] += [2]
    elif change=='duplicate':
        review['complete_indices'] += [0]
    elif change=='omitted':
        review['excluded'] = {}
    elif change=='overlap':
        review['complete_indices'] += [1]
    elif change=='not_approved':
        review['training_approved'] = False
    else:
        review['queue_sha256'] = 'changed'
    write_json(rp,review)
    with pytest.raises(ValueError):
        check_case(review_case)


def test_review_rejects_changed_geometry_even_with_refreshed_queue_hash(review_case):
    *_,queue,review,qp,rp = review_case
    queue = copy.deepcopy(queue)
    queue['frames'][0]['truth'][0]['box'] = [1,2,3,4]
    write_json(qp,queue)
    review['queue_sha256'] = sha256(qp)
    write_json(rp,review)
    with pytest.raises(ValueError,match='annotations'):
        check_case(review_case)


def test_baseline_comparison_ignores_only_latency():
    from scripts.summarize_dense_background_trial import check_baseline
    a = dict(id=1,source='frame.jpg',frame_sha256='hash',truth=[],counts={},
             detections=[dict(score=.8,accepted=True,latency_ms=1)])
    b = copy.deepcopy(a)
    b['detections'][0]['latency_ms'] = 2
    check_baseline(a,b)
    b['detections'][0]['score'] = .9
    with pytest.raises(ValueError,match='output'):
        check_baseline(a,b)
