import numpy as np
import pytest

from scripts.balloon_two_task_projection import project_numpy,project_tensor


@pytest.mark.parametrize('a,b',[( [1.,0.],[1.,2.]),([1.,0.],[-1.,1.]),([1.,0.],[-1.,0.]),([0.,0.],[1.,2.])])
def test_projection_tensor_matches_independent_numpy(a,b):
    tf=pytest.importorskip('tensorflow')
    actual=project_tensor(tf.constant(a),tf.constant(b)).numpy()
    np.testing.assert_allclose(actual,project_numpy(a,b),atol=1e-6)
    assert np.dot(actual,a)>=-1e-6
    assert np.dot(actual,b)>=-1e-6


def test_conflict_removed_without_changing_cooperative_sum():
    np.testing.assert_array_equal(project_numpy([1.,0.],[2.,1.]),[3.,1.])
    np.testing.assert_allclose(project_numpy([1.,0.],[-1.,1.]),[.5,1.5])
    with pytest.raises(ValueError):
        project_numpy([np.nan],[1.])


def test_no_single_seed_promotion():
    from scripts.audit_balloon_gradient_projection import paired_gate
    results={n:dict(training_scene_gate_pass=True) for n in ('sum47','sum48','project47','project48')}
    assert paired_gate(results)
    results['project48']['training_scene_gate_pass']=False
    assert not paired_gate(results)
    del results['sum47']
    with pytest.raises(ValueError):
        paired_gate(results)
