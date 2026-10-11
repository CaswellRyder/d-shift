from scripts.research_confident_containment import select


def observation(box, score, label='blue_balloon', accepted=True):
    return dict(box=box, crop_box=box, score=score, label=label, accepted=accepted)


def test_stronger_child_replaces_weak_parent_without_changing_scores():
    rows = [observation([0, 0, 100, 100], .85), observation([10, 10, 40, 40], .99)]
    output = select(rows)
    assert [r['accepted'] for r in output] == [False, True]
    assert [r['score'] for r in output] == [.85, .99]
    assert all(r['accepted'] for r in rows)


def test_stronger_parent_keeps_part_suppressed_and_ties_prefer_parent():
    for scores in ((.99, .85), (.99, .99)):
        rows = [observation([0, 0, 100, 100], scores[0]), observation([10, 10, 40, 40], scores[1])]
        assert [r['accepted'] for r in select(rows)] == [True, False]


def test_different_colors_remain_and_background_never_resurrects():
    rows = [observation([0, 0, 100, 100], .85), observation([10, 10, 40, 40], .99, 'red_balloon'),
            observation([70, 70, 90, 90], .999, 'background', False)]
    assert [r['accepted'] for r in select(rows)] == [True, True, False]


def test_ordinary_duplicate_suppression_is_not_reversed():
    rows = [observation([0, 0, 100, 100], .99), observation([1, 1, 99, 99], .98)]
    output = select(rows)
    assert [r['accepted'] for r in output] == [True, False]
    assert output[1]['rejection_reason'] == 'confidence_balloon_duplicate'
