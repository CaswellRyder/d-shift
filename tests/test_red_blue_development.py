from scripts.evaluate_red_blue_development import detection_counts, metrics


def test_matching_counts_duplicate_and_wrong_color_and_missing_target():
    truth = [dict(label="red_balloon", box=[0, 0, 10, 10]),
             dict(label="blue_balloon", box=[20, 0, 30, 10])]
    predictions = [dict(label="red_balloon", box=[0, 0, 10, 10], score=.99, accepted=True),
                   dict(label="red_balloon", box=[0, 0, 10, 10], score=.98, accepted=True),
                   dict(label="red_balloon", box=[20, 0, 30, 10], score=.97, accepted=True)]
    assert detection_counts(truth, predictions) == {"red_balloon": dict(tp=1, fp=2, fn=0),
                                                   "blue_balloon": dict(tp=0, fp=0, fn=1)}


def test_rejected_label_cannot_count_as_detection():
    truth = [dict(label="red_balloon", box=[0, 0, 10, 10])]
    counts = detection_counts(truth, [dict(truth[0], score=.4, accepted=False)])
    assert counts["red_balloon"] == dict(tp=0, fp=0, fn=1)
    result = metrics(counts)
    assert result["red_balloon"]["recall"] == 0
    assert result["red_balloon"]["precision"] is None
    assert result["blue_balloon"]["recall"] is None
