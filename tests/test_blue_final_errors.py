import pytest

from scripts.audit_blue_final_errors import missed_stages


@pytest.mark.parametrize('rows,reason', [
    ([], 'no_geometric_proposal'),
    ([dict(label='background', accepted=False)], 'localized_but_wrong_label'),
    ([dict(label='blue_balloon', accepted=False)], 'correct_label_below_threshold'),
    ([dict(label='blue_balloon', accepted=False, raw_accepted=True)], 'suppressed_correct_detection'),
])
def test_miss_stage_partition(rows, reason):
    frame = dict(truth=[dict(label='blue_balloon', box=[0, 0, 10, 10])],
                 detections=[dict(box=[0, 0, 10, 10], score=.9, **r) for r in rows])
    coverage, misses = missed_stages(frame)
    assert misses == {('blue_balloon', reason): 1}
    assert coverage['blue_balloon'] == bool(rows)


def test_one_proposal_cannot_match_two_targets():
    target = dict(label='blue_balloon', box=[0, 0, 10, 10])
    frame = dict(truth=[target, target], detections=[dict(target, score=.9, accepted=True)])
    coverage, misses = missed_stages(frame)
    assert coverage['blue_balloon'] == 2  # Optimistic geometric bound, not attainable recall.
    assert misses == {('blue_balloon', 'one_to_one_matching_competition'): 1}


def test_success_does_not_count_as_miss():
    target = dict(label='red_balloon', box=[0, 0, 10, 10])
    coverage, misses = missed_stages(dict(truth=[target], detections=[dict(target, score=.9, accepted=True)]))
    assert coverage['red_balloon'] == 1
    assert not misses
