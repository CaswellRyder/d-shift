import pytest

from scripts.openimages_annotation_window import complete_window

HEADER = b'ImageID,LabelName,XMin,YMin,XMax,YMax\n'


def row(i, cls='/m/01j51'):
    return f'{i:016x},{cls},0,0,1,1\n'.encode()


def test_both_edge_groups_removed_including_multiple_annotations():
    raw = b'clipped-first-line\n'+row(1)+row(1)+row(2)+row(2)+row(3)+b'clipped-tail'
    groups = complete_window(raw, HEADER)
    assert list(groups) == ['0000000000000002']
    assert len(groups['0000000000000002']) == 2


def test_nonballoon_edge_rows_still_identify_incomplete_groups():
    raw = b'partial\n'+row(1, '/m/other')+row(1)+row(2)+row(3)+row(3, '/m/other')
    assert list(complete_window(raw, HEADER)) == ['0000000000000002']


@pytest.mark.parametrize('raw', [b'partial', b'partial\n', b'partial\n'+row(1), b'partial\n'+row(1)+row(1)])
def test_no_complete_interior_group_yields_no_admission(raw):
    assert complete_window(raw, HEADER) == {}


@pytest.mark.parametrize('raw', [b'partial\n'+row(2)+row(1),
    b'partial\nnot-a-valid-id,/m/01j51,0,0,1,1\n',
    b'partial\n0000000000000001,/m/01j51,1,0,0,1\n',
    b'partial\n0000000000000001,/m/01j51,0\n'])
def test_unsafe_records_fail_closed(raw):
    with pytest.raises(ValueError):
        complete_window(raw, HEADER)


def test_missing_header_columns_rejected():
    with pytest.raises(ValueError):
        complete_window(b'partial\n'+row(1)+row(2)+row(3), b'ImageID,LabelName\n')


@pytest.mark.parametrize('changes', [{'start_byte':0}, {'scan_bytes':63 << 20}, {'limit':97}])
def test_acquisition_budgets_rejected_before_network(tmp_path, changes):
    from scripts.acquire_blue_balloon_window import acquire
    with pytest.raises(ValueError):
        acquire(tmp_path/'unused', **changes)
    assert not (tmp_path/'unused').exists()
