import numpy as np
import pytest

from scripts.balloon_rb_swap import swap, select_sources


def test_swap_is_involution_and_preserves_green_and_background_label():
    pixels = np.arange(36, dtype=np.float32).reshape(3, 2, 2, 3)
    labels = np.array([0, 1, 2])
    before = pixels.copy()
    transformed, mapped = swap(pixels, labels)
    np.testing.assert_array_equal(mapped, [0, 2, 1])
    np.testing.assert_array_equal(transformed[..., 1], pixels[..., 1])
    restored, restored_labels = swap(transformed, mapped)
    np.testing.assert_array_equal(restored, pixels)
    np.testing.assert_array_equal(restored_labels, labels)
    np.testing.assert_array_equal(before, pixels)


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'labels', 'range'])
def test_invalid_augmentation_input_rejected(bad):
    x, y = np.zeros((1, 2, 2, 3), np.float32), np.array([2])
    if bad == 'shape':
        x = x[0]
    elif bad == 'dtype':
        x = x.astype(np.float64)
    elif bad == 'labels':
        y[0] = -1
    else:
        x[0, 0, 0, 0] = 256
    with pytest.raises(ValueError):
        swap(x, y)


def fixture():
    return [dict(source_image=f'train/source{i//3}', sha256=str(i), label=['background','red_balloon','blue_balloon'][i%3],
                 view='original', split='train') for i in range(12)]


def test_source_cap_dedup_and_foreground_balance():
    rows = fixture()
    rows.append(dict(rows[0]))
    selected = select_sources(rows)
    chosen = [rows[i] for i in selected]
    assert sum(r['label']=='red_balloon' for r in chosen) == sum(r['label']=='blue_balloon' for r in chosen)
    assert all(sum(r['source_image']==s for r in chosen) <= 2 for s in {r['source_image'] for r in chosen})
    assert len({r['sha256'] for r in chosen}) == len(chosen)


@pytest.mark.parametrize('bad', ['split', 'img', 'conflict'])
def test_unsafe_sources_rejected(bad):
    rows = fixture()
    if bad == 'split':
        rows[0]['split'] = 'test'
    elif bad == 'img':
        rows[0]['source_image'] = 'train/IMG_heldout'
    else:
        rows.append(dict(rows[0], label='red_balloon'))
    with pytest.raises(ValueError):
        select_sources(rows)
