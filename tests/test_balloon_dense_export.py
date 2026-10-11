import numpy as np
import pytest

from scripts.balloon_dense_runtime import prepare_input,DenseProposals
from scripts.export_balloon_dense_proposals import proposal_parity
from dtr.data import sha256,write_json


def test_dense_preprocessing_keeps_rgb_and_uint8_range():
    rgb=np.zeros((240,320,3),np.uint8)
    rgb[:]=[12,34,250]
    prepared=prepare_input(rgb)
    assert prepared.shape==(1,120,160,3) and prepared.dtype==np.float32
    np.testing.assert_array_equal(prepared[0,0,0],[12,34,250])
    with pytest.raises(ValueError):
        prepare_input(rgb.astype(np.float32))


@pytest.mark.parametrize('change', ['count','crop','geometry','group','score'])
def test_export_rejects_decoded_geometry_or_decision_changes(change):
    original=dict(box=[1.,2.,11.,12.],crop_box=[0,0,13,14],color_group=1,proposal_score=.9)
    actual=dict(original)
    if change=='crop':
        actual['crop_box']=[0,0,14,14]
    if change=='geometry':
        actual['box']=[2.,2.,11.,12.]
    if change=='group':
        actual['color_group']=0
    if change=='score':
        actual['proposal_score']=.8
    with pytest.raises(ValueError):
        proposal_parity([] if change=='count' else [actual],[original])
    proposal_parity([original],[original])


def test_dense_runtime_rejects_changed_hash_before_interpreter(tmp_path):
    path=tmp_path/'proposal.tflite'
    path.write_bytes(b'fixture')
    write_json(path.with_suffix('.json'),dict(kind='balloon_dense_proposal_research',sha256='bad'))
    with pytest.raises(ValueError,match='identity'):
        DenseProposals(path)
    assert sha256(path)!='bad'
