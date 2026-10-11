import pytest

from dtr.data import sha256, write_json
from scripts.build_balloon_color_control_bundle import prepare


@pytest.mark.parametrize('bad', ['source_scope', 'candidate_mode', 'hash', 'seed', 'alias'])
def test_bundle_rejects_invalid_inputs_before_writing(tmp_path, bad):
    reference = tmp_path/'reference'
    reference.mkdir()
    write_json(reference/'bundle.json', dict(files={}, deployment_approved=False, test_evaluated=False))
    write_json(reference/'inputs.json', dict(split='test' if bad == 'source_scope' else 'development_validation',
        default_model='old', models={'fragment-color-47': {}} if bad == 'alias' else {}, frames=[]))
    model = tmp_path/'candidate.tflite'
    model.write_bytes(b'fixture; must not open interpreter')
    write_json(model.with_suffix('.json'), dict(sha256='invalid' if bad == 'hash' else sha256(model),
        deployment_approved=False, task='balloon', classes=['background','red_balloon','blue_balloon'],
        training=dict(mode='fragment' if bad == 'candidate_mode' else 'control',
                      seed=48 if bad == 'seed' else 47)))
    output = tmp_path/'output'
    with pytest.raises(ValueError):
        prepare(reference, output, 'unused.so', model)
    assert not output.exists()
