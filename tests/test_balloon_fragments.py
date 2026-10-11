import numpy as np
import pytest

from scripts.train_balloon_fragments import expanded_student,fragment_anchor,loss_function
from scripts.balloon_fragment_runtime import prediction


def test_fragment_requires_a_small_piece_of_a_reviewed_blue_balloon():
    target=dict(label='blue_balloon',box=[10,10,110,110])
    assert fragment_anchor(dict(crop_box=[20,20,30,30]),dict(truth=[target]))==target
    for targets,crop in (([target],[10,10,110,110]),([], [20,20,30,30]),
                          ([dict(target,label='red_balloon')],[20,20,30,30]),
                          ([target,target],[20,20,30,30]),([target],[105,105,120,120])):
        with pytest.raises(ValueError,match='parent'):
            fragment_anchor(dict(crop_box=crop),dict(truth=targets))


def test_extended_head_preserves_initial_color_logits_and_has_129_new_parameters():
    import keras
    from dtr.models import research_student
    parent=research_student(3,64,'separable_context')
    model=expanded_student(parent)
    x=np.ones((2,64,64,3),np.float32)
    original=parent(x,training=False).numpy()
    new=model(x,training=False).numpy()
    np.testing.assert_array_equal(new[:,:3],original)
    np.testing.assert_allclose(new[:,3],original[:,2]-6,atol=1e-6)
    assert model.count_params()==parent.count_params()+129
    assert keras.Model(model.input,model.output).output_shape==(None,4)


def test_fragment_rows_have_no_teacher_preservation_term():
    import tensorflow as tf
    target=np.array([[0,0,0,1,1,2,3,0],[0,0,1,0,1,2,3,1]],np.float32)
    prediction=tf.constant([[1.,2.,3.,4.],[1.,2.,3.,4.]])
    loss=loss_function()
    before=loss(tf.constant(target),prediction).numpy()
    target[:,4:7]=[7.,-7.,-7.]
    after=loss(tf.constant(target),prediction).numpy()
    assert before[0]==after[0] and before[1]!=after[1]


def test_fragment_and_background_cannot_be_accepted_targets():
    assert not prediction([0,0,0,20])['accepted']
    assert prediction([0,0,0,20])['fragment_rejected']
    assert not prediction([20,0,0,0])['accepted']
    assert prediction([0,20,0,0])['accepted']
    assert prediction([0,0,20,0])['accepted']
    assert not prediction([0,0,2,2])['accepted']
    for values in ([0,0,0],[0,0,0,float('nan')]):
        with pytest.raises(ValueError):
            prediction(values)


def test_standard_runtime_rejects_fragment_models_even_in_research_mode(tmp_path):
    from dtr.data import write_json,sha256
    from dtr.runtime import Predictor
    path=tmp_path/'fragment.tflite'
    path.write_bytes(b'fixture; no interpreter should open this')
    write_json(path.with_suffix('.json'),dict(kind='balloon_fragment_research',sha256=sha256(path)))
    with pytest.raises(ValueError,match='explicit'):
        Predictor(path,allow_unvalidated=True)


@pytest.mark.parametrize('mode,state,tampered', [('fragment','complete',False),
    ('control','running',False), ('control','complete',True)])
def test_color_export_rejects_fragment_incomplete_or_changed_provenance(tmp_path,mode,state,tampered):
    from dtr.data import write_json,sha256
    from scripts.export_balloon_fragment_control import export
    from scripts.train_balloon_fragments import CLASSES
    write_json(tmp_path/'provenance.json',dict(mode=mode,classes=CLASSES))
    write_json(tmp_path/'status.json',dict(state=state))
    write_json(tmp_path/'fragment.fp32.json',dict(provenance_sha256=(
        'changed' if tampered else sha256(tmp_path/'provenance.json'))))
    with pytest.raises(ValueError,match='Only completed'):
        export(tmp_path)
    assert not (tmp_path/'color.keras').exists()
