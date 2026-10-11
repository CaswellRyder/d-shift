from scripts.audit_blue_object_losses import failure_reason


def test_separates_classification_confidence_and_selection():
    target=dict(label='blue_balloon',box=[0,0,10,10])
    r=dict(box=[0,0,10,10],label='background',raw_accepted=False,accepted=False)
    assert failure_reason(target,[])=='no_localized_proposal'
    assert failure_reason(target,[r])=='class_changed'
    r['label']='blue_balloon'
    assert failure_reason(target,[r])=='confidence_below_threshold'
    r['raw_accepted']=True
    assert failure_reason(target,[r])=='selection_suppressed'
    r['accepted']=True
    assert failure_reason(target,[r])=='matching_assignment'
