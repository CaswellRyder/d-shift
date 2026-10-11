import pytest

from scripts.acquire_blue_balloon_candidates import BALLOON, eligible_annotations, unseen_selection


def test_candidates_exclude_all_previously_attempted_ids_and_bound_downloads():
    groups = {f'{i:016x}': [] for i in range(230)}
    excluded = set(list(groups)[:144])
    selected = unseen_selection(groups, excluded)
    assert len(selected) == 86 and not set(selected) & excluded
    assert selected == unseen_selection(dict(reversed(list(groups.items()))), excluded)
    assert len(unseen_selection(groups, [])) == 96
    for limit in (0, 97, True, 1.5):
        with pytest.raises(ValueError):
            unseen_selection(groups, [], limit)


def test_screen_does_not_use_depictions_groups_or_other_objects():
    original = dict(LabelName=BALLOON, IsDepiction='0', IsGroupOf='0')
    rows = [original, dict(original, IsDepiction='1'), dict(original, IsGroupOf='1'),
            dict(original, LabelName='/m/0bt_c3')]
    assert eligible_annotations(rows) == [original]
