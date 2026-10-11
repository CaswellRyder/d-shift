from scripts.audit_balloon_context_distillation import scene_gate, false_acceptances


def test_actual_scene_gate_not_perfect_clutter_crop_fit():
    effect=dict(lost_targets=[],count_delta={'red_balloon':dict(tp=0,fp=-1,fn=0),
                                            'blue_balloon':dict(tp=1,fp=0,fn=-1)})
    assert scene_gate(effect,7,3)
    assert scene_gate(effect,7,7)
    assert not scene_gate(effect,7,8)
    assert not scene_gate(dict(effect,lost_targets=['one-target']),7,0)
    effect['count_delta']['red_balloon']['fp']=1
    assert not scene_gate(effect,7,0)


def test_background_argmax_and_rejection_are_different():
    assert false_acceptances([[0,4,0],[0,.1,0],[4,0,0]])==1
