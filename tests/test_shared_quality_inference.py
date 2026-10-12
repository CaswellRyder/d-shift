from types import SimpleNamespace

import numpy as np
import pytest

import scripts.shared_quality_inference as runtime


def test_batches_only_accepted_rows_and_preserves_parent(monkeypatch):
    row = dict(box=[0., 0., 16., 16.], label='red_balloon', score=.9,
               proposal_origin='mser', accepted=True)
    parent = [row, dict(row, proposal_origin='learned'), dict(row, accepted=False)]
    monkeypatch.setattr(runtime, 'classify', lambda *args: [])
    monkeypatch.setattr(runtime, 'decode', lambda *args: [])
    monkeypatch.setattr(runtime, 'as_direct', lambda *args: [])
    monkeypatch.setattr(runtime, 'raw_direct', lambda *args: [])
    monkeypatch.setattr(runtime, 'merge', lambda *args: parent)
    seen = []

    def head(x, training):
        seen.append(x.shape)
        assert training is False
        return SimpleNamespace(numpy=lambda: np.array([[.2], [.8]], np.float32))

    old, new, count = runtime.infer(None, [], None, np.zeros((15, 20, 24), np.float32),
                                   None, head, np.zeros(33), np.ones(33))
    assert old == parent and count == 2 and seen == [(2, 33)]
    assert [r['accepted'] for r in new] == [False, True, False]
    assert old[0]['accepted'] is True


def test_invalid_normalization_fails_before_inference():
    for scale in (np.zeros(33), np.full(33, np.nan), np.ones(32)):
        with pytest.raises(ValueError):
            runtime.infer(None, [], None, None, None, None, np.zeros(33), scale)
