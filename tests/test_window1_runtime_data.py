import copy

import pytest

from scripts.window1_runtime_data import choose, EASY_IDS


def fixture():
    photos = [5, 5, 9, 10, 10, 13, 20, 20, 25]
    rows = [dict(id=i, frame_id=f, source_image=f'openimages/train/{f}',
                 source_sha256=f'photo{f}', sha256=f'crop{i}', split='train',
                 label='red_balloon' if i in (18, 34, 65) else 'blue_balloon')
            for i, f in zip(EASY_IDS, photos, strict=True)]
    hard = [dict(rows[pos], id=i, sha256=f'hard{i}') for i, pos in enumerate((2, 6))]
    return rows, hard


def test_only_two_same_photo_crops_replaced_without_mutation():
    rows, hard = fixture()
    original = copy.deepcopy((rows, hard))
    assert choose(rows, hard, 'control') == rows
    actual = choose(rows, hard, 'hard')
    assert [i for i, (a, b) in enumerate(zip(rows, actual, strict=True)) if a != b] == [2, 6]
    assert [r['label'] for r in actual] == [r['label'] for r in rows]
    assert (rows, hard) == original


@pytest.mark.parametrize('mutation', ['source', 'source_hash', 'frame', 'label', 'split', 'img',
                                    'duplicate', 'arm', 'order', 'extra', 'balance', 'easy_label'])
def test_unsafe_replacement_rejected(mutation):
    rows, hard = fixture()
    arm = 'hard'
    if mutation == 'source':
        hard[0]['source_image'] = 'openimages/train/elsewhere'
    elif mutation == 'source_hash':
        hard[0]['source_sha256'] = 'changed'
    elif mutation == 'frame':
        hard[0]['frame_id'] = 100
    elif mutation == 'label':
        hard[0]['label'] = 'red_balloon'
    elif mutation == 'split':
        hard[0]['split'] = 'test'
    elif mutation == 'img':
        rows[0]['source_image'] = 'openimages/train/IMG_reserved'
    elif mutation == 'duplicate':
        hard[0]['sha256'] = rows[0]['sha256']
    elif mutation == 'arm':
        arm = 'unknown'
    elif mutation == 'order':
        rows.reverse()
    elif mutation == 'extra':
        hard.append(dict(hard[0], id=2))
    elif mutation == 'balance':
        rows[5]['source_image'] = rows[0]['source_image']
    else:
        rows[0]['label'] = 'red_balloon'
    with pytest.raises(ValueError):
        choose(rows, hard, arm)


def test_duplicate_easy_pixels_rejected():
    rows, hard = fixture()
    rows[1]['sha256'] = rows[0]['sha256']
    with pytest.raises(ValueError):
        choose(rows, hard, 'control')
