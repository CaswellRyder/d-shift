from copy import deepcopy

from scripts.compare_current_blue_search import gate


def test_blue_gain_cannot_hide_more_false_alarms_or_lost_red_targets():
    old = {c: dict(tp=9, fp=1, fn=1) for c in ('red_balloon', 'blue_balloon')}
    new = deepcopy(old)
    new['blue_balloon'].update(tp=10, fn=0)
    assert gate(old, new)
    new['blue_balloon']['fp'] = 2
    assert not gate(old, new)
    new['blue_balloon']['fp'] = 1
    new['red_balloon'].update(tp=8, fn=2)
    assert not gate(old, new)
