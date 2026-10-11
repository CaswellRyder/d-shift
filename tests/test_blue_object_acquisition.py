import numpy as np
import pytest

from scripts.acquire_blue_objects import object_groups, choose, blue_fraction


def test_excludes_balloon_group_and_incomplete_tail():
    header = 'ImageID,LabelName,XMin,YMin,XMax,YMax,IsDepiction,IsGroupOf\n'
    raw = header+''.join(f'{i:016x},{label},0,0,1,1,0,0\n' for i,label in
        [(1,'/m/01n4qj'),(2,'/m/01n4qj'),(2,'/m/01j51'),(3,'/m/01n4qj')])
    assert set(object_groups(raw.encode()))=={'0000000000000001'}


def test_selection_is_bounded_deterministic_and_excludes_known_sources():
    groups = {f'{i:016x}':[dict(LabelName='/m/01n4qj')] for i in range(100)}
    first = choose(groups,{'0000000000000000'})
    assert len(first)==32
    assert '0000000000000000' not in first
    assert first==choose(dict(reversed(list(groups.items()))),{'0000000000000000'})
    with pytest.raises(ValueError):
        choose(groups,set(),33)


def test_blue_screen_uses_object_not_unrelated_pixels():
    rgb = np.zeros((240,320,3),np.uint8)
    rgb[:,:160,2] = 255
    annotations = [dict(LabelName='/m/0bt_c3',XMin=.5,YMin=0,XMax=1,YMax=1)]
    assert blue_fraction(rgb,annotations)[0]['fraction']==0
    annotations[0]['XMin']=0
    assert blue_fraction(rgb,annotations)[0]['fraction']==.5
