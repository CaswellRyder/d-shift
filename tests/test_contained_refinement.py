from copy import deepcopy

import pytest

from scripts.research_contained_refinement import eligible, refine


def row(label='blue_balloon',accepted=True,box=None):
    return dict(label=label,accepted=accepted,box=box or [20,20,60,60],
                crop_box=[16,16,64,64],score=.95,scores=[.02,.03,.95])


def proposal(box=None,score=.9):
    return dict(color_group=1,proposal_score=score,box=box or [25,25,55,55])


def test_refinement_changes_only_blue_geometry_and_retains_original_input():
    rows=[row(),row('red_balloon',box=[25,25,55,55]),row(accepted=False)]
    before=deepcopy(rows)
    # A red detection in the same place must protect the occupied location.
    assert refine(rows,[proposal()])==rows
    result=refine(rows[:1]+[row('red_balloon',box=[100,100,130,130])],[proposal()])
    assert result[0]['box']==[25,25,55,55]
    assert result[0]['pre_refinement_box']==before[0]['box']
    for k in ('crop_box','label','score','scores','accepted'):
        assert result[0][k]==before[0][k]
    assert rows==before
    assert refine([row(accepted=False)],[proposal()])==[row(accepted=False)]


def test_constraints_reject_enlargement_low_confidence_and_distant_box():
    parent=[20,20,60,60]
    for p in (proposal([10,10,70,70]),proposal(score=.49),proposal([50,20,80,50]),proposal([29,29,39,39])):
        assert not eligible(parent,p)
    with pytest.raises(ValueError):
        eligible(parent,proposal([0,0,float('nan'),30]))


def test_refinement_does_not_create_detections():
    assert refine([],[proposal()])==[]
    rows=[row(box=[x,20,x+40,60]) for x in (20,100,200)]
    proposals=[proposal([x+5,25,x+35,55]) for x in (20,100,200)]
    result=refine(rows,proposals)
    assert len(result)==len(rows)==3
    assert sum('pre_refinement_box' in r for r in result)==2
