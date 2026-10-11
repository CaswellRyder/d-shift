import hashlib
import io
import zipfile

import cv2
import numpy as np
from PIL import Image
import pytest

from scripts.research_balloon_resolution import high_proposals, native_frame


def test_native_search_maps_geometry_and_caps_calls():
    rgb = np.zeros((480,640,3), np.uint8)
    cv2.circle(rgb, (200,180), 60, (0,0,240), -1)
    proposals = high_proposals(rgb)
    assert 0 < len(proposals) <= 12
    assert any(r['color_group'] == 1 and r['box'] == [70,60,131,121] for r in proposals)
    assert all(0 <= r['crop_box'][0] < r['crop_box'][2] <= 320
               and 0 <= r['crop_box'][1] < r['crop_box'][3] <= 240 for r in proposals)


@pytest.mark.parametrize('shape,dtype', [((240,320,3),np.uint8), ((480,640,3),np.float32),
                                        ((480,640),np.uint8)])
def test_rejects_wrong_resolution_or_encoding(shape,dtype):
    with pytest.raises(ValueError, match='640x480'):
        high_proposals(np.zeros(shape,dtype))


def test_native_loader_checks_source_pixels_and_reviewed_resize():
    pixels = np.random.default_rng(2).integers(0,256,(480,640,3),dtype=np.uint8)
    png, data = io.BytesIO(), io.BytesIO()
    Image.fromarray(pixels).save(png,format='PNG')
    with zipfile.ZipFile(data,'w') as archive:
        archive.writestr('train/frame_0001.png', png.getvalue())
    expected = cv2.resize(pixels,(320,240),interpolation=cv2.INTER_AREA)
    allowed = {'train/frame_0001.png':hashlib.sha256(pixels.tobytes()).hexdigest()}
    with zipfile.ZipFile(data) as archive:
        np.testing.assert_array_equal(native_frame(archive,'frame_0001.png',expected,allowed),pixels)
        with pytest.raises(ValueError, match='reviewed baseline'):
            native_frame(archive,'frame_0001.png',np.zeros_like(expected),allowed)
        with pytest.raises(ValueError, match='pixels changed'):
            native_frame(archive,'frame_0001.png',expected,{'train/frame_0001.png':'wrong'})


def test_reserved_or_unadmitted_sources_rejected_before_decode():
    for source in ('IMG_1234.jpg','train/IMG_1234.jpg','frame_missing.jpg'):
        with pytest.raises(ValueError):
            native_frame(None,source,None,{})


def test_supplement_preserves_baseline_and_call_budget():
    from scripts.summarize_balloon_resolution import bounded_supplement
    base = [dict(box=[0,0,10,10],color_group=0)]
    high = [base[0], dict(box=[20,0,30,10],color_group=1),
            dict(box=[40,0,50,10],color_group=1), dict(box=[60,0,70,10],color_group=1)]
    assert bounded_supplement(base,high) == base+high[1:3]
    assert bounded_supplement(base*12,high) == base*12
