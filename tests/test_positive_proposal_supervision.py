from scripts.positive_proposal_supervision import annotation_label


def test_labels_follow_annotations_not_a_prediction():
    truth = [dict(label='blue_balloon', box=[0, 0, 20, 20])]
    assert annotation_label([0, 0, 20, 20], truth)['label'] == 'blue_balloon'


def test_partial_or_distant_boxes_are_not_pseudo_negatives():
    truth = [dict(label='red_balloon', box=[0, 0, 20, 20])]
    assert annotation_label([0, 0, 5, 5], truth) is None
    assert annotation_label([30, 30, 40, 40], truth) is None
    assert annotation_label([0, 0, 20, 20], []) is None


def test_opposite_color_overlap_is_ambiguous():
    truth = [dict(label='red_balloon', box=[0, 0, 20, 20]),
             dict(label='blue_balloon', box=[10, 0, 30, 20])]
    assert annotation_label([0, 0, 20, 20], truth) is None


def test_same_color_multiple_objects_can_agree():
    truth = [dict(label='blue_balloon', box=[0, 0, 20, 20]),
             dict(label='blue_balloon', box=[10, 0, 30, 20])]
    assert annotation_label([0, 0, 20, 20], truth)['label'] == 'blue_balloon'
