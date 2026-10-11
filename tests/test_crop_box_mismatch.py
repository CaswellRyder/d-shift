from scripts.audit_crop_box_mismatch import crop_geometry


def test_distinguishes_box_localization_from_classifier_context():
    truth = [dict(label='blue_balloon',box=[10,10,20,20])]
    detection = dict(label='blue_balloon',box=[0,10,9,20],crop_box=[0,8,12,22])
    result = crop_geometry(detection,truth)
    assert result['same_color_in_crop']
    assert not result['same_color_in_box']
    assert result['max_same_color_target_fraction']==.2


def test_empty_crop_overlap_does_not_invent_targets():
    result = crop_geometry(dict(label='red_balloon',box=[0,0,5,5],crop_box=[0,0,6,6]),
        [dict(label='blue_balloon',box=[2,2,5,5])])
    assert not result['same_color_in_crop']
    assert result['other_color_in_crop']
    assert result['max_same_color_target_fraction']==0
