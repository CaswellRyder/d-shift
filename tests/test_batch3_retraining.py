from collections import Counter

import numpy as np
import pytest

from scripts.train_batch3_retraining import calculate, choose_controls


def test_control_selection_is_deterministic_unique_original_and_class_matched():
    rows = [dict(sha256=str(i), label='blue_balloon' if i < 10 else 'red_balloon',
                 preserve=True, view='original') for i in range(12)]
    extra = [dict(label='blue_balloon')]*6+[dict(label='red_balloon')]
    chosen = choose_controls(rows+rows, extra)
    assert chosen == choose_controls(rows+rows, extra)
    assert len({r['sha256'] for r in chosen}) == 7
    assert Counter(r['label'] for r in chosen) == Counter(r['label'] for r in extra)
    with pytest.raises(ValueError):
        choose_controls([dict(r, view='horizontal_flip') for r in rows], extra)


def test_training_gate_rejects_any_lost_parent_row_or_unlearned_probe():
    labels = np.array([0, 1, 2])
    before = np.eye(3, dtype=np.float32)*8
    probes = before[[1, 2]]
    expected = np.array([1, 2])
    assert calculate(before, before, labels, probes, expected)['training_gate_pass']
    after = before.copy()
    after[0] = before[2]
    r = calculate(before, after, labels, probes, expected)
    assert not r['training_gate_pass'] and r['lost_old_rows'] == [0]
    assert r['false_acceptances']['blue_balloon'] == dict(before=0, after=1)
    assert not calculate(before, before, labels, probes[::-1], expected)['training_gate_pass']


def test_background_correctness_and_positive_confidence_are_distinct():
    v = np.array([[.1, 0., 0.], [0., .1, 0.]])
    labels = np.array([0, 1])
    r = calculate(v, v, labels, np.array([[0., 8., 0.]]), np.array([1]))
    assert r['old_correct_before'] == 1
