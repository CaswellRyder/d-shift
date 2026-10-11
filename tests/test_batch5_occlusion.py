import copy

import pytest

from scripts.batch5_occlusion_data import choose, new_scene_gate


def fixture():
    ids = [3, 4, 8, 52, 55, 56, 68]
    sources = [1, 1, 4, 15, 17, 17, 25]
    rows = [dict(id=i, frame_id=s, source_image=f'openimages/train/{s}', sha256=str(i),
                 label='blue_balloon', split='train') for i, s in zip(ids, sources, strict=True)]
    hard = dict(rows[5], id=0, sha256='hard')
    p = dict(arms=['control', 'hard'], easy_ids=ids, replace_easy_id=56, hard_id=0)
    return rows, hard, p


def test_only_one_same_photo_crop_replaced_without_mutation():
    rows, hard, p = fixture()
    original = copy.deepcopy(rows)
    assert choose(rows, hard, 'control', p) == rows
    actual = choose(rows, hard, 'hard', p)
    assert [r['id'] for r in actual] == [3, 4, 8, 52, 55, 0, 68]
    assert rows == original


@pytest.mark.parametrize('mutation', ['source', 'label', 'split', 'img', 'duplicate', 'arm', 'order'])
def test_invalid_substitution_rejected(mutation):
    rows, hard, p = fixture()
    arm = 'hard'
    if mutation == 'source':
        hard['source_image'] = 'different'
    elif mutation == 'label':
        hard['label'] = 'background'
    elif mutation == 'split':
        hard['split'] = 'test'
    elif mutation == 'img':
        rows[0]['source_image'] = 'train/IMG_reserved'
    elif mutation == 'duplicate':
        hard['sha256'] = rows[0]['sha256']
    elif mutation == 'arm':
        arm = 'unknown'
    else:
        rows.reverse()
    with pytest.raises(ValueError):
        choose(rows, hard, arm, p)


def test_gate_requires_every_target_no_false_positive_and_hard_acceptance():
    result = dict(metrics=dict(red_balloon=dict(tp=0, fp=0, fn=0), blue_balloon=dict(tp=8, fp=0, fn=0)))
    assert new_scene_gate(result, True)
    assert not new_scene_gate(result, False)
    for label, key in [('red_balloon', 'fp'), ('blue_balloon', 'fp'), ('blue_balloon', 'fn')]:
        changed = copy.deepcopy(result)
        changed['metrics'][label][key] += 1
        assert not new_scene_gate(changed, True)
