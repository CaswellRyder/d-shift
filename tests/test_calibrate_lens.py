import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dtr.lens import LensModel

spec = importlib.util.spec_from_file_location(
    'calibrate_lens', Path(__file__).parents[1] / 'scripts' / 'calibrate_lens.py')
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def render_boards(folder, count):
    """Ray-cast a physical checkerboard through an ideal fisheye, then antialias.

    Rendering uses inverse ray/plane intersections, independently of the calibration
    detector. Images live exclusively in pytest's temporary directory.
    """
    lens = LensModel.ideal_equidistant(1280, 960, axis='horizontal')
    yy, xx = np.mgrid[:960, :1280]
    rays = lens.rays(np.column_stack((xx.ravel(), yy.ravel())))
    rng = np.random.default_rng(42)
    for i in range(count):
        rot, _ = cv2.Rodrigues(rng.uniform(-.5, .5, 3))
        # Board has 8 x 6 squares, 7 x 5 inner corners, 4cm squares.
        translation = np.array([rng.uniform(-.16, .16), rng.uniform(-.11, .11),
                                rng.uniform(.30, .55)])
        normal = rot[:, 2]
        denominator = rays @ normal
        distance = np.divide(translation @ normal, denominator,
                             out=np.full(len(rays), -1.), where=denominator > 0)
        local = (rays * distance[:, None] - translation) @ rot
        col = np.floor((local[:, 0] + .16) / .04).astype(int)
        row = np.floor((local[:, 1] + .12) / .04).astype(int)
        inside = (col >= 0) & (col < 8) & (row >= 0) & (row < 6) & (distance > 0)
        pixels = np.full(len(rays), 180, np.uint8)
        pixels[inside] = (((col[inside] + row[inside]) % 2) * 255).astype(np.uint8)
        image = cv2.resize(pixels.reshape(960, 1280), (320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / f'board-{i:02d}.png'), cv2.flip(image, -1))


def test_synthetic_checkerboard_calibration(tmp_path, capsys):
    images = tmp_path / 'images'
    images.mkdir()
    render_boards(images, 32)
    output = tmp_path / 'lens.json'
    result = calibration.calibrate(images, 7, 5, .04, output)
    ideal = LensModel.ideal_equidistant(axis='horizontal')
    assert abs(result['K'][0][0] / ideal.K[0, 0] - 1) < .03
    assert abs(result['K'][1][1] / ideal.K[1, 1] - 1) < .03
    assert result['images_used'] >= 10
    assert result['flipped'] is True
    assert json.loads(output.read_text()) == result
    assert LensModel.from_calibration(output).calibrated
    text = capsys.readouterr().out
    assert 'RMS' in text and 'horizontal' in text and 'diagonal' in text


def test_refuses_fewer_than_ten_boards(tmp_path):
    render_boards(tmp_path, 3)
    output = tmp_path / 'calibration.json'
    with pytest.raises(ValueError, match='at least 10'):
        calibration.calibrate(tmp_path, 7, 5, .04, output)
    assert not output.exists()


def test_ideal_fov():
    assert calibration.measured_fov(LensModel.ideal_equidistant(axis='horizontal')) == pytest.approx([120, 90, 150])
