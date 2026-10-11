from collections import Counter

import pytest

from scripts.window2_retraining_data import choose_controls
from scripts.reviewed_balloon_window import inventory,load_batch


def rows():
    return [dict(sha256=str(i),source_image=f'train/{i//4}',label='blue_balloon',preserve=True,view='original') for i in range(16)]


def test_controls_unique_source_capped_and_deterministic():
    a=choose_controls(rows(),[dict(label='blue_balloon')]*6)
    assert a==choose_controls(rows(),[dict(label='blue_balloon')]*6)
    assert len({r['sha256'] for r in a})==6
    assert max(Counter(r['source_image'] for r in a).values())<=2


def test_controls_fail_closed_on_source_shortage():
    with pytest.raises(ValueError):
        choose_controls(rows(),[dict(label='blue_balloon')]*9)


def test_window_inventory_rejects_duplicate_training_views():
    r=rows()[:2]
    review=dict(admitted_crops=2,admitted_photographs=1,label_counts={'blue_balloon':2},max_crops_per_photo=2)
    inventory(r,review)
    r[1]=dict(r[0])
    with pytest.raises(ValueError):
        inventory(r,review)


def test_review_hash_checked_before_source_access(tmp_path):
    path=tmp_path/'review.json'
    path.write_text('{}')
    with pytest.raises(ValueError,match='Sealed review'):
        load_batch(path,'0'*64)


def test_crop_recovery_cannot_override_scene_regression():
    from scripts.audit_window2_retraining import training_gate
    effect=dict(lost_targets=[],gained_targets=[],count_delta={'blue':dict(tp=0,fp=0,fn=0)})
    assert training_gate(effect,effect,[7,2],[True]*15)
    assert not training_gate(effect,effect,[7,2],[True]*14+[False])
    bad=dict(effect,lost_targets=['target'])
    assert not training_gate(effect,bad,[7,2],[True]*15)
    assert not training_gate(effect,effect,[7,8],[True]*15)
    with pytest.raises(ValueError):
        training_gate(effect,effect,[7,2],[])
