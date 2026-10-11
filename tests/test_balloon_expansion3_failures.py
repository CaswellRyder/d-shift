from scripts.audit_balloon_expansion3_failures import explain,suspect


def test_stage_attribution_separates_localization_classification_and_selection():
    target=dict(label='blue_balloon',box=[0,0,20,20])
    hit=dict(box=[0,0,20,20],label='blue_balloon',raw_accepted=True,accepted=True)
    assert explain(target,[])=='proposal_localization'
    assert explain(target,[dict(hit,box=[0,0,5,5])])=='proposal_localization'
    assert explain(target,[dict(hit,label='red_balloon')])=='classification'
    assert explain(target,[dict(hit,raw_accepted=False,accepted=False)])=='classification'
    assert explain(target,[dict(hit,accepted=False)])=='selection'
    assert explain(target,[hit])=='covered'


def test_suspects_are_only_a_review_queue_not_negative_labels():
    target=dict(label='blue_balloon',box=[0,0,20,20])
    found=[dict(target,accepted=True),dict(target,label='red_balloon',accepted=True),
           dict(target,box=[50,50,80,80],accepted=True),dict(target,accepted=False)]
    assert suspect(found,[target])==found[1:3]
    assert suspect(found,[])==found[:3]
