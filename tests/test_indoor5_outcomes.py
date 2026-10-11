import pytest

from scripts.audit_indoor5_outcomes import compare_frames, reviewed_indices, REVIEWED
from scripts.evaluate_red_blue_development import detection_counts


def frame(boxes):
    truth = [dict(label='blue_balloon', box=[x, 0, x + 10, 10]) for x in (0, 20)]
    ds = [dict(label='blue_balloon', box=b, score=.9, accepted=True, raw_accepted=True) for b in boxes]
    return dict(id=1, source='fixture', frame_sha256='fixture', truth=truth,
                detections=ds, counts=detection_counts(truth, ds))


def test_equal_counts_do_not_hide_lost_targets():
    before, after = frame([[0, 0, 10, 10]]), frame([[20, 0, 30, 10]])
    assert before['counts'] == after['counts']
    result = compare_frames([after], [before])
    assert result['lost_parent_targets'][0]['target']['box'] == [0, 0, 10, 10]
    assert result['gained_targets'][0]['target']['box'] == [20, 0, 30, 10]
    assert result['colors']['blue_balloon']['geometric_oracle_covered'] == 1


def test_changed_comparison_identity_rejected():
    f = frame([])
    with pytest.raises(ValueError, match='Different'):
        compare_frames([dict(f, source='other')], [f])


def test_visual_review_cannot_expand_to_unsaved_frame():
    assert reviewed_indices(dict(saved_sample_indices=list(REVIEWED))) == REVIEWED
    with pytest.raises(ValueError, match='visual review'):
        reviewed_indices(dict(saved_sample_indices=list(range(8))))
