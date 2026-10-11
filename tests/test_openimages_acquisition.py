import pytest

from scripts.acquire_openimages_balloons import complete_boxes, acquire

HEADER = b'ImageID,LabelName,XMin,YMin,XMax,YMax\n'


def test_range_parser_drops_partial_tail_and_last_image_group():
    raw = HEADER+b'0000000000000001,/m/01j51,0,0,1,1\n0000000000000002,/m/01j51,0,0,1,1\npartial'
    assert list(complete_boxes(raw)) == ['0000000000000001']


def test_range_parser_does_not_accept_other_classes_or_invalid_boxes():
    raw = HEADER+b'0000000000000001,/m/other,0,0,1,1\n0000000000000002,/m/01j51,0,0,1,1\n'
    assert complete_boxes(raw) == {}
    with pytest.raises(ValueError,match='box'):
        complete_boxes(HEADER+b'0000000000000001,/m/01j51,1,0,0,1\n')


def test_acquisition_rejects_unbounded_work_before_network(tmp_path):
    for limit,size in ((0,1 << 20),(97,1 << 20),(48,129 << 20)):
        with pytest.raises(ValueError):
            acquire(tmp_path/'absent',size,limit)
    with pytest.raises(FileExistsError):
        acquire(tmp_path)


def test_license_requires_photo_license_anchor_not_incidental_text():
    from scripts.verify_openimages_licenses import LicenseLinks, cc_by_2
    parser = LicenseLinks()
    parser.feed('<p>https://creativecommons.org/licenses/by/2.0/</p>'
                '<a href="https://creativecommons.org/licenses/by/2.0/">unrelated</a>'
                '<a rel="license noopener" href="https://creativecommons.org/licenses/by/2.0/deed.en">license</a>')
    assert parser.links == ['https://creativecommons.org/licenses/by/2.0/deed.en']
    assert cc_by_2(parser.links[0])
    assert not cc_by_2('https://creativecommons.org.evil.test/licenses/by/2.0/')
    assert not cc_by_2('https://creativecommons.org/licenses/by-nc/2.0/')


def test_review_source_path_cannot_escape_root(tmp_path):
    from scripts.prepare_openimages_balloon_review import checked_path
    with pytest.raises(ValueError):
        checked_path(tmp_path,'../outside','not-a-digest')


def test_training_labels_require_complete_nonoverlapping_review():
    from scripts.openimages_balloon_training import review_labels
    review = dict(training_approved=True,split='train',deployment_approved=False,
        labels_by_class={'background':[0],'red_balloon':[1],'blue_balloon':[2]},exclude=[3])
    assert review_labels(review,4) == {0:'background',1:'red_balloon',2:'blue_balloon'}
    for changes in ({'split':'test'},{'exclude':[2,3]},{'exclude':[]},{'training_approved':False}):
        with pytest.raises(ValueError):
            review_labels(dict(review,**changes),4)


def test_public_trial_cannot_silently_change_old_training_defaults(tmp_path):
    from scripts.train_balloon_reviewed_clutter import train
    for kwargs in ({'openimages':True},{'openimages':True,'epochs':24},
                   {'hard_weight':4},{'epochs':24},
                   {'openimages':True,'expansion4':True,'hard_weight':4,'epochs':24}):
        with pytest.raises(ValueError):
            train('admit',47,tmp_path/'unused',**kwargs)
    assert not (tmp_path/'unused').exists()


def test_large_background_crop_cannot_swallow_tiny_balloon_at_low_iou():
    from scripts.openimages_balloon_training import safe_background
    row = dict(label='background',kind='unmatched_proposal',crop_box=[0,0,320,240])
    frame = dict(boxes=[[240,105,248,113]])
    assert not safe_background(row,frame)
    assert safe_background(dict(row,crop_box=[0,0,30,30]),frame)
    # Explicitly reviewed non-target source objects are different from unmatched clutter.
    assert safe_background(dict(row,kind='source_balloon_box'),frame)
