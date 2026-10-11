import numpy as np
import pytest

from scripts.audit_balloon_source_concentration import summarize


def row(source, label='blue_balloon'):
    return dict(source_image=source, label=label, split='train', view='original')


def test_repetition_is_not_new_source_coverage():
    rows = [row('train/a')]*3 + [row('train/b')]
    result = summarize(rows, np.array([[1], [1], [1], [2]], dtype=np.float32))
    assert result['rows'] == 4
    assert result['named_sources'] == result['unique_tensors'] == 2
    assert result['repeated_tensor_rows'] == 2
    assert result['concentration_equivalent_sources'] == 1.6


def test_exact_pixel_conflicting_labels_are_reported():
    result = summarize([row('train/a'), row('train/b', 'background')], np.ones((2, 1)))
    assert len(result['exact_tensor_label_conflicts']) == 1


@pytest.mark.parametrize('bad', [dict(split='test'), dict(source_image='train/IMG_excluded')])
def test_evaluation_family_never_enters_inventory(bad):
    with pytest.raises(ValueError):
        summarize([{**row('train/a'), **bad}], np.zeros((1, 1)))
