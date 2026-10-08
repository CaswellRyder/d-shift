from scripts.audit_balloon_scene_errors import attribute


def test_errors_separate_search_classification_threshold_and_suppression():
    target = dict(label="blue_balloon", box=[0, 0, 10, 10])
    detection = dict(target, score=.9, accepted=True)
    def reason(ds):
        return attribute(dict(truth=[target], detections=ds))[0]["reason"]
    assert reason([]) == "no_localized_proposal"
    assert reason([dict(detection, box=[20, 20, 30, 30])]) == "no_localized_proposal"
    assert reason([dict(detection, label="background", accepted=False)]) == "wrong_class_on_localized_proposals"
    assert reason([dict(detection, score=.7, accepted=False)]) == "correct_class_below_threshold"
    assert reason([dict(detection, accepted=False, suppressed=True)]) == "suppression_involved"
    assert reason([detection]) == "detected"


def test_one_detection_cannot_count_as_two_targets():
    target = dict(label="blue_balloon", box=[0, 0, 10, 10])
    result = attribute(dict(truth=[target, target], detections=[dict(target, score=.9, accepted=True)]))
    assert sorted(r["reason"] for r in result) == ["detected", "one_to_one_assignment_conflict"]
