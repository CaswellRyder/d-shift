from scripts.prepare_blue_false_alarm_review import select_review
from scripts.openimages_balloon_training import review_labels
from dtr.data import read_json


def test_review_selection_is_bounded_deterministic_and_source_capped():
    pool = [dict(source=f's{i // 3}', crop_box=[i, 0, i + 1, 1],
                 mining_prediction=dict(score=.9)) for i in range(90)]
    selected = select_review(pool)
    assert len(selected) == 48
    assert selected == select_review(reversed(pool))
    assert all(sum(r['source'] == s['source'] for r in selected) <= 2 for s in selected)


def test_checked_in_labels_cover_every_candidate_exactly_once():
    review = read_json('configs/blue-false-alarm-review-20261011.json')
    labels = review_labels(review, 48)
    assert len(labels) == 39
    assert set(labels.values()) == {'background'}
    assert {int(k) for k in review['exclusion_reasons']} == set(review['exclude'])
    assert all(review['exclusion_reasons'].values())
