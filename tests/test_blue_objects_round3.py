from scripts.acquire_blue_objects_round3 import previous_ids
from scripts.acquire_blue_objects import choose, CLASSES


def test_round3_excludes_previously_retained_and_screened_out_ids():
    old = previous_ids([dict(images=[dict(image_id='a')], screened_out=[dict(image_id='b')])])
    label = next(iter(CLASSES))
    assert choose({i: [dict(LabelName=label)] for i in ('a', 'b', 'c')}, old) == {'c': label}
