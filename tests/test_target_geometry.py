import json
import time
from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from dtr.contracts import TargetReport
from dtr.lens import LensModel
from dtr.target_geometry import CameraMount, TargetGeometry


def project(lens, xyz):
    return cv2.fisheye.projectPoints(np.asarray(xyz, np.float64).reshape(1, -1, 3),
                                    np.zeros(3), np.zeros(3), lens.K, lens.D)[0].reshape(-1, 2)


def target_box(lens, distance, bearing, diameter, sphere):
    angle = np.radians(bearing)
    ray = np.array([np.sin(angle), 0, np.cos(angle)])
    right = np.array([np.cos(angle), 0, -np.sin(angle)])
    down = np.array([0, 1, 0])
    if sphere:
        # True tangent circle on a sphere, not a planar circle at center depth.
        radius = diameter / 2
        silhouette_center = ray * (distance - radius**2 / distance)
        silhouette_radius = radius * np.sqrt(1 - (radius / distance)**2)
        theta = np.linspace(0, 2 * np.pi, 4096)
        xyz = (silhouette_center + silhouette_radius *
               (np.cos(theta)[:, None] * right + np.sin(theta)[:, None] * down))
    else:
        # Face-on square; sample edges because fisheye edges are curved.
        t = np.linspace(-0.5, 0.5, 200)
        xy = np.concatenate([np.column_stack((t, np.full_like(t, s))) for s in (-.5, .5)] +
                            [np.column_stack((np.full_like(t, s), t)) for s in (-.5, .5)])
        xyz = ray * distance + diameter * (xy[:, :1] * right + xy[:, 1:] * down)
    pixels = project(lens, xyz)
    return [*pixels.min(axis=0), *pixels.max(axis=0)]


@pytest.mark.parametrize('distance', [1, 2, 4, 6])
@pytest.mark.parametrize('bearing', [0, -30, 30, -50, 50])
@pytest.mark.parametrize('label,width,sphere', [('green_balloon', .4, True),
                                               ('orange_square', 1.168, False)])
def test_projected_geometry(distance, bearing, label, width, sphere):
    lens = LensModel.ideal_equidistant(axis='horizontal')
    target = TargetGeometry(lens).target(label, .9,
                                        target_box(lens, distance, bearing, width, sphere))
    assert abs(target.range_m / distance - 1) < (.08 if abs(bearing) == 50 else .03)
    assert abs(target.bearing_deg - bearing) < 1


def test_distortion_scaling_and_inverse(tmp_path):
    lens = LensModel([[155, 0, 158], [0, 153, 121], [0, 0, 1]],
                     [.04, -.015, .003, -.0002], (320, 240))
    path = tmp_path / 'lens.json'
    data = dict(K=lens.K.tolist(), D=lens.D.ravel().tolist(), image_size=[320, 240],
                rms=.1, flipped=True)
    path.write_text(json.dumps(data))
    loaded = LensModel.from_calibration(path, 640, 480)
    rays = np.array([[np.tan(np.radians(a)), .05, 1] for a in (-60, -30, 0, 30, 60)])
    expected = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    assert np.allclose(loaded.rays(project(lens, rays) * 2), expected, atol=1e-9)
    assert loaded.calibrated
    box = target_box(lens, 4, 30, .4, True)
    result = TargetGeometry(LensModel.from_calibration(path)).target('green_balloon', 1, box)
    assert abs(result.range_m / 4 - 1) < .03
    data['flipped'] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='flipped'):
        LensModel.from_calibration(path)


def test_mount_signs_errors_clipping_contracts():
    lens = LensModel.ideal_equidistant(axis='horizontal')
    geometry = TargetGeometry(lens, mount=CameraMount(pitch_deg=-10))
    target = geometry.target('orange_square', .8, [140, 100, 180, 140])
    assert target.climb_err_deg == pytest.approx(-10)
    assert target.yaw_err_deg == pytest.approx(0)
    assert TargetGeometry(lens, mount=CameraMount(yaw_deg=15)).target(
        'orange_square', 1, [140, 100, 180, 140]).yaw_err_deg == pytest.approx(15)
    geometry = TargetGeometry(lens)
    assert geometry.target('green_balloon', 1, [180, 80, 200, 100]).yaw_err_deg > 0
    assert geometry.target('green_balloon', 1, [180, 80, 200, 100]).climb_err_deg > 0
    near = geometry.target('green_balloon', 1, [140, 100, 180, 140])
    far = geometry.target('green_balloon', 1, [155, 115, 165, 125])
    assert far.range_err_m > near.range_err_m
    for box in ([1, 50, 20, 70], [20, 1, 40, 20], [300, 50, 319, 70], [20, 220, 40, 239]):
        clipped = geometry.target('green_balloon', 1, box)
        assert clipped.clipped and clipped.range_is_upper_bound
    shifted = TargetGeometry(lens, mount=CameraMount(translation_m=(1, 2, 3))).target(
        'green_balloon', 1, [140, 100, 180, 140])
    assert np.allclose(np.array(shifted.body_xyz_m) - near.body_xyz_m, [1, 2, 3])
    assert shifted.range_m == pytest.approx(np.linalg.norm(shifted.body_xyz_m))
    report = TargetReport(1, 'balloon', near, None, True)
    json.dumps(report.to_dict(), allow_nan=False)
    json.dumps(near.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        near.score = 0
    with pytest.raises(FrozenInstanceError):
        report.ready = False
    assert near.size_calibrated and not near.lens_calibrated  # Balloon size is measured.
    assert target.size_calibrated


def test_grid_and_invalid_inputs():
    lens = LensModel.ideal_equidistant(axis='horizontal')
    bearing, elevation = lens.grid(step=40)
    assert bearing.shape == elevation.shape == (6, 8)
    assert bearing[3, 4] == pytest.approx(0)
    assert lens.rays([]).shape == (0, 3)
    with pytest.raises(ValueError):
        lens.rays([[np.nan, 0]])
    with pytest.raises(ValueError):
        lens.grid(0)
    with pytest.raises(ValueError):
        TargetGeometry(lens).target('green_balloon', 1, [2, 2, 1, 1])


def test_geometry_performance():
    geometry = TargetGeometry(LensModel.ideal_equidistant(axis='horizontal'))
    for _ in range(10):
        geometry.target('green_balloon', 1, [130, 90, 190, 150])
    start = time.perf_counter()
    for _ in range(300):
        for label in ('green_balloon', 'purple_balloon', 'orange_square'):
            geometry.target(label, 1, [130, 90, 190, 150])
    elapsed_ms = (time.perf_counter() - start) * 1000 / 300
    print(f'Geometry for 3 targets: {elapsed_ms:.4f} ms mean; Pi: unknown')
    assert elapsed_ms < 1


@pytest.mark.parametrize('distance', [.3, .4, .6])
def test_close_sphere_uses_exact_tangent_range(distance):
    # Capture happens up close, where the small-angle tangent formula read ~25% short.
    lens = LensModel.ideal_equidistant(axis='horizontal')
    target = TargetGeometry(lens).target('green_balloon', .9,
                                         target_box(lens, distance, 0, .4, True))
    assert target.range_m == pytest.approx(distance, rel=.03)


def test_rolled_square_reads_closer():
    # Documented limit: a square rolled 45 degrees spans its diagonal, reading ~1/sqrt(2).
    lens = LensModel.ideal_equidistant(axis='horizontal')
    target = TargetGeometry(lens).target('orange_square', .9,
                                         [128.89, 88.89, 191.11, 151.11])
    assert target.range_m == pytest.approx(4 / np.sqrt(2), rel=.03)


def test_default_lens_is_120_diagonal():
    from scripts.calibrate_lens import measured_fov
    horizontal, vertical, diagonal = measured_fov(LensModel.ideal_equidistant())
    assert diagonal == pytest.approx(120, abs=.01)
    assert horizontal == pytest.approx(96, abs=.1)  # We measured ~95 side to side.
    assert vertical == pytest.approx(72, abs=.1)


@pytest.mark.parametrize('distance', [2, 4, 6])
@pytest.mark.parametrize('bearing', [0, -25, 25])
@pytest.mark.parametrize('label,width,sphere', [('green_balloon', .4, True),
                                               ('orange_square', 1.168, False)])
def test_projected_geometry_default_lens(distance, bearing, label, width, sphere):
    lens = LensModel.ideal_equidistant()
    target = TargetGeometry(lens).target(label, .9,
                                         target_box(lens, distance, bearing, width, sphere))
    assert target.range_m == pytest.approx(distance, rel=.03)
    assert abs(target.bearing_deg - bearing) < 1


@pytest.mark.parametrize('color', ['orange', 'yellow'])
@pytest.mark.parametrize('shape,width,height', [('circle', 1.130, 1.130),
                                               ('square', 1.168, 1.168),
                                               ('triangle', 1.749, 1.515)])
@pytest.mark.parametrize('distance', [2., 4., 6.])
@pytest.mark.parametrize('width_scale,height_scale', [(1., 1.), (.6, 1.), (1., .6)])
def test_flat_goal_range_uses_minimum_axis(color, shape, width, height, distance,
                                         width_scale, height_scale):
    lens = LensModel.ideal_equidistant()
    # Mid-edge rays for a flat upright face; yaw/pitch shorten one extent only.
    rays = [[-width * width_scale / 2, 0, distance], [width * width_scale / 2, 0, distance],
            [0, -height * height_scale / 2, distance], [0, height * height_scale / 2, distance]]
    pixels = project(lens, rays)
    box = [pixels[0, 0], pixels[2, 1], pixels[1, 0], pixels[3, 1]]
    result = TargetGeometry(lens).target(f'{color}_{shape}', .9, box)
    assert result.range_m == pytest.approx(distance, rel=1e-6)
    assert result.size_calibrated and result.range_err_m > 0


@pytest.mark.parametrize('height', [None, 0, -1, float('nan'), float('inf')])
def test_flat_goal_requires_valid_height(tmp_path, height):
    size = dict(width_m=1.749, shape='triangle', calibrated=True)
    if height is not None:
        size['height_m'] = height
    path = tmp_path / 'sizes.json'
    path.write_text(json.dumps({'yellow_triangle': size}))
    with pytest.raises(ValueError, match='height_m'):
        TargetGeometry(LensModel.ideal_equidistant(), sizes_path=path)
