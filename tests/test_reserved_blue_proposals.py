import pytest

from scripts.audit_balloon_dense_proposals import supplement
from scripts.research_reserved_blue_proposals import reserved


def row(i, color=1):
    return dict(box=[i*10, 0, i*10+5, 5], color_group=color)


def test_saturated_pixels_leave_room_for_two_unique_learned_blue():
    base = [row(i) for i in range(12)]
    assert reserved(base, [row(15), row(16), row(17)]) == base[:10]+[row(15), row(16)]
    assert len(base) == 12


def test_do_not_displace_pixels_for_duplicates_or_red():
    base = [row(i) for i in range(12)]
    assert reserved(base, [row(11), row(16, color=0)]) == base
    assert reserved(base, [row(15), row(15)]) == base[:11]+[row(15)]


def test_unsaturated_policy_unchanged_and_bounded():
    for count in range(11):
        base, learned = [row(i) for i in range(count)], [row(15), row(16)]
        assert reserved(base, learned) == supplement(base, learned)
    base = [row(i) for i in range(11)]
    assert reserved(base, []) == base
    assert reserved(base, [row(15)]) == base+[row(15)]
    with pytest.raises(ValueError):
        reserved([row(i) for i in range(13)], [])
