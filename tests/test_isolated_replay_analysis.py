import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'replay_analysis', Path(__file__).parents[1] / 'scripts/analyze_isolated_replay.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(frame=0, boxes=None):
    return dict(source='a', clip=0, frame=frame, result_age_ms=50,
                boxes=[] if boxes is None else boxes,
                truth=[dict(label='yellow_square', box=[0, 0, 10, 10])])


def test_duplicate_predictions_cannot_match_twice():
    scored = analysis.score([row(boxes=[[0, 0, 10, 10]] * 2)])
    assert (scored['tp'], scored['fp'], scored['fn']) == (1, 1, 0)
    assert scored['size_buckets']['under16']['recall'] == 1
    assert scored['size_buckets']['64plus']['recall'] is None


def test_matching_reassigns_for_maximum_cardinality():
    assert analysis.matched_truth([[0, 0, 15, 10], [0, 0, 10, 10]],
                                  [[0, 0, 10, 10], [5, 0, 15, 10]]) == {0, 1}


def test_pairing_excludes_skips_and_rejects_truth_mismatch():
    p, c = analysis.pair_rows([row(0), row(1)], [row(1), row(2)])
    assert len(p) == len(c) == 1
    assert p[0]['frame'] == 1
    altered = row(1)
    altered['truth'] = []
    with pytest.raises(ValueError, match='ground truth'):
        analysis.pair_rows([row(1)], [altered])
    with pytest.raises(ValueError, match='Duplicate'):
        analysis.pair_rows([row(1), row(1)], [])


def test_reacquisition_includes_skips_and_missing_results():
    result = analysis.reacquisition([row(15, [[0, 0, 10, 10]])])[0]
    assert result['delivery_delay_ms'] == 250
    missed = analysis.reacquisition([row(15)])[0]
    assert missed['delivery_delay_ms'] is None
    assert not missed['reacquired']


def test_scheduling_counts_unprocessed_tail():
    record = row(0)
    record['skipped'] = 0
    result = analysis.scheduling([record])
    assert result['total_skipped'] == 29
    assert result['skipped_before_processed_frames'] == 0
