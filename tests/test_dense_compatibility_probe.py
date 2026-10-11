import json

import numpy as np
import pytest

from dtr.data import sha256, write_json
from scripts.pi_dense_compatibility_probe import verified
from scripts.research_dense_supported_selection import supported_selection
from scripts.verify_dense_compatibility_probe import verify


def fixture(tmp_path):
    root, results = tmp_path/'bundle', tmp_path/'results'
    root.mkdir()
    results.mkdir()
    values = np.zeros((1,15,20,6), np.float32)
    values[:,:,:,:2] = -80
    np.save(root/'host-47.npy', values)
    np.save(results/'predictions-47.npy', values)
    (root/'model.tflite').write_bytes(b'identity-fixture-not-an-executable-model')
    raw = [dict(box=[0,0,100,100], crop_box=[0,0,100,100], accepted=True,
                label='blue_balloon', score=.99, scores=[0,0,.99])]
    write_json(root/'golden.json', [dict(raw=raw,selected={'47':supported_selection(raw,[])})])
    write_json(root/'manifest.json', dict(
        models={'47':'model.tflite'},frames=1,
        files={p.name:sha256(p) for p in root.iterdir()},
        policy_sha256=sha256('scripts/research_dense_supported_selection.py'),
        decoder_sha256=sha256('scripts/audit_balloon_dense_proposals.py')))
    write_json(results/'report.json',dict(machine='test-fixture',
        manifest_sha256=sha256(root/'manifest.json'),models={'47':dict(calls=1,
            model_sha256=sha256(root/'model.tflite'),
            output_sha256=sha256(results/'predictions-47.npy'))}))
    return root, results


def test_exact_parity_is_scoped_not_flight_proof(tmp_path):
    root, results = fixture(tmp_path)
    result = verify(root, results)
    assert result['models']['47']['exact_selection_parity']
    assert not result['camera_tested'] and not result['deployment_approved']
    assert not result['isolated_speed_measured']


@pytest.mark.parametrize('change', ['nonfinite','drift','wrong_shape','selection'])
def test_changed_native_outputs_fail_closed(tmp_path, change):
    root, results = fixture(tmp_path)
    path = results/'predictions-47.npy'
    value = np.load(path)
    if change=='nonfinite':
        value[0,0,0,0] = np.nan
    elif change=='drift':
        value[0,0,0,0] += .003
    elif change=='wrong_shape':
        value = value[:,:,:19,:]
    else:
        golden = json.loads((root/'golden.json').read_text())
        golden[0]['selected']['47'][0]['accepted'] = False
        write_json(root/'golden.json',golden)
        manifest = json.loads((root/'manifest.json').read_text())
        manifest['files']['golden.json'] = sha256(root/'golden.json')
        write_json(root/'manifest.json',manifest)
    np.save(path,value)
    report = json.loads((results/'report.json').read_text())
    report['models']['47']['output_sha256'] = sha256(path)
    report['manifest_sha256'] = sha256(root/'manifest.json')
    write_json(results/'report.json',report)
    with pytest.raises(ValueError):
        verify(root,results)


def test_bundle_hash_and_path_escape_are_rejected(tmp_path):
    root, _ = fixture(tmp_path)
    (root/'model.tflite').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        verified(root)
    outside = tmp_path/'outside'
    outside.write_bytes(b'outside')
    write_json(root/'manifest.json',dict(files={'../outside':sha256(outside)}))
    with pytest.raises(ValueError,match='escaped'):
        verified(root)
