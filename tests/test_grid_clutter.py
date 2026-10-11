from scripts.prepare_grid_clutter_review import guarded_background


def test_negative_requires_whole_crop_outside_all_expanded_targets():
    truth = [dict(box=[100, 100, 120, 120], label='blue_balloon')]
    assert guarded_background([0, 0, 20, 20], truth)
    assert not guarded_background([90, 90, 95, 95], truth)
    assert not guarded_background([0, 0, 105, 105], truth)
    assert guarded_background([0, 0, 84, 84], truth)


def test_blue_surface_preserves_unmasked_pixels_and_changes_only_red():
    import numpy as np
    from scripts.grid_clutter_data import blue_surface
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [90, 90, 90], [0, 0, 255]]], np.uint8)
    changed, fraction = blue_surface(rgb)
    assert fraction == .25
    assert changed[0, 0, 2] > changed[0, 0, 0]
    np.testing.assert_array_equal(changed[0, 1:], rgb[0, 1:])
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])


def test_blue_surface_rejects_non_rgb_or_non_byte_data():
    import numpy as np
    import pytest
    from scripts.grid_clutter_data import blue_surface
    for bad in (np.zeros((3, 3), np.uint8), np.zeros((3, 3, 4), np.uint8), np.zeros((3, 3, 3), np.float32)):
        with pytest.raises(ValueError, match='RGB uint8'):
            blue_surface(bad)


def test_fit_counts_do_not_count_hard_rows_as_ordinary_accuracy():
    import numpy as np
    from scripts.train_balloon_grid_clutter import fit_counts
    result = fit_counts(np.array([[2, 0, 0], [0, 0, 2], [0, 2, 0]]),
                        np.array([0, 1, 1]), np.array([True, True, False]))
    assert result == dict(background=dict(correct=1, total=1), red_balloon=dict(correct=0, total=1),
                          blue_balloon=dict(correct=0, total=0))


def test_paired_rows_preserve_ordinary_and_apply_same_view_schedule(monkeypatch):
    import scripts.grid_clutter_data as module
    ordinary = [dict(label='background', path=f'{i}.png', sha256=str(i)) for i in range(20)]
    extra = [dict(label='background', file=f'new{i}.png', sha256=f'new{i}') for i in range(11)]
    fragments = [dict(path='fragment.png', sha256='positive')]
    monkeypatch.setattr(module, 'checked_data', lambda: (ordinary, fragments))
    monkeypatch.setattr(module, 'checked_extra', lambda: extra)
    control, _ = module.data('control')
    admit, _ = module.data('admit')
    assert control[:24] == admit[:24]
    assert len(control) == len(admit) == 68
    assert all(r['preserve'] for r in control[:24])
    assert all(not r['preserve'] for r in admit[24:])
    assert [r['view'] for r in control] == [r['view'] for r in admit]
    assert len({r['sha256'] for r in control[24:]}) == 11
    assert len({r['sha256'] for r in admit[24:]}) == 11
    assert {r['label'] for r in control[24:] + admit[24:]} == {'background'}


def test_nonregression_rejects_precision_gained_by_losing_targets():
    from scripts.summarize_balloon_grid_clutter import nonregression
    reference = dict(red_balloon=dict(tp=70, fp=4, fn=1), blue_balloon=dict(tp=67, fp=24, fn=4))
    candidate = dict(red_balloon=dict(tp=61, fp=0, fn=10), blue_balloon=dict(tp=62, fp=13, fn=9))
    assert not nonregression(candidate, reference)
    assert nonregression(reference, reference)
    better = dict(red_balloon=dict(tp=70, fp=3, fn=1), blue_balloon=dict(tp=67, fp=13, fn=4))
    assert nonregression(better, reference)
    changed_truth = dict(better, red_balloon=dict(tp=70, fp=3, fn=0))
    assert not nonregression(changed_truth, reference)
