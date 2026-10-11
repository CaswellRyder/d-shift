import copy
import pytest

from scripts.research_dense_supported_selection import supported_selection,candidate_pairs


def rows():
    def row(box):
        return dict(box=box,crop_box=box,label='red_balloon',accepted=True,score=.99,scores=[.01,.99,0])
    return [row([0,0,100,100]),row([40,40,60,60])]


def test_supported_child_replaces_oversized_parent_without_mutation():
    found=rows()
    before=copy.deepcopy(found)
    selected=supported_selection(found,[dict(box=[40,40,60,60],color_group=0,proposal_score=.8)])
    assert found==before
    assert not selected[0]['accepted'] and selected[1]['accepted']
    assert selected[0]['raw_accepted'] and selected[0]['suppression_reason']=='dense_supported_child'
    assert [r['box'] for r in selected]==[r['box'] for r in found]


def test_weak_wrong_color_or_rejected_child_cannot_replace_parent():
    evidence=dict(box=[40,40,60,60],color_group=0,proposal_score=.8)
    for variant in ([],[dict(evidence,proposal_score=.49)],[dict(evidence,color_group=1)],
                    [dict(evidence,box=[0,0,100,100])]):
        selected=supported_selection(rows(),variant)
        assert selected[0]['accepted'] and not selected[1]['accepted']
    found=rows()
    found[1]['accepted']=False
    selected=supported_selection(found,[evidence])
    assert selected[0]['accepted'] and not selected[1]['accepted']


def test_no_nested_pair_gate_is_exact_for_any_dense_evidence():
    found=rows()
    found[1]['box']=[120,120,140,140]
    assert candidate_pairs(found)==[]
    evidence=[dict(box=[120,120,140,140],color_group=0,proposal_score=1.)]
    assert supported_selection(found,evidence)==supported_selection(found,[])
    assert candidate_pairs(rows())==[(0,1)]


def test_gate_receipt_rejects_changed_acceptances_but_allows_timing():
    from scripts.summarize_dense_supported_selection import same_output
    before=dict(id=0,source='frame.jpg',truth=[],detections=rows())
    after=copy.deepcopy(before)
    after['detections'][0]['latency_ms']=123
    same_output(before,after)
    after['detections'][0]['accepted']=False
    with pytest.raises(ValueError,match='gating'):
        same_output(before,after)
