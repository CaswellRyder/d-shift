from scripts.audit_balloon_false_positives import classify_errors


def detection(box, label='blue_balloon', score=.99):
    return dict(box=box, label=label, score=score, accepted=True)


def test_duplicate_wrong_color_mislocalization_and_background_are_separate():
    truth = [dict(box=[0, 0, 10, 10], label='blue_balloon'), dict(box=[50, 0, 60, 10], label='red_balloon')]
    original = detection([0, 0, 10, 10])
    frame = dict(truth=truth, detections=[original, detection([0, 0, 10, 10], score=.98),
        detection([50, 0, 60, 10], score=.97), detection([5, 0, 20, 10], score=.96),
        detection([100, 100, 120, 120], score=.95)])
    errors = classify_errors(frame, dict(detections=[original]))
    assert [r['reason'] for r in errors] == ['duplicate_localized_same_color', 'localized_wrong_color',
        'overlaps_target_below_matching_iou', 'no_annotated_target_overlap']
    assert errors[0]['origin'] == 'baseline_search'
    assert errors[1]['origin'] == 'added_localizer'


def test_rejected_detection_is_not_false_positive():
    row = detection([0, 0, 10, 10])
    row['accepted'] = False
    assert classify_errors(dict(truth=[], detections=[row]), dict(detections=[])) == []
