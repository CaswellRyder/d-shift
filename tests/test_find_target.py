import cv2
import numpy as np
import pytest

from dtr.find_target import FakeFindTarget, FindTarget, best_track
from dtr.flight import Track
from dtr.lens import LensModel


class StubWorker:
    def __init__(self):
        self.state = 'stopped'
        self.result = None
        self.submissions = []

    @property
    def ready(self):
        return self.state == 'idle'

    def start(self):
        self.state = 'idle'

    def wait_ready(self, timeout):
        assert timeout > 0

    def close(self):
        self.state = 'stopped'

    def poll(self):
        result, self.result = self.result, None
        return result

    def submit(self, rgb, frame_time, phase):
        self.submissions.append((frame_time, phase))

    def detection(self, phase, stamp, labels):
        self.result = dict(phase=phase, frame_time=stamp,
                           detections=[dict(label=label, score=.9, box=box) for label, box in labels])


def frame():
    hsv = np.zeros((240, 320, 3), np.uint8)
    for hue, box in [(60, (30, 60, 60, 90)), (10, (100, 80, 140, 120)),
                     (30, (200, 80, 250, 130))]:
        x1, y1, x2, y2 = box
        hsv[y1:y2, x1:x2] = (hue, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_find_target_phase_and_color_hits():
    worker = StubWorker()
    with FindTarget(worker=worker) as finder:
        finder.wait_ready(1)
        rgb = frame()
        assert finder.balloon(rgb, 0).target is None
        worker.detection('balloon', 0, [('green_balloon', [30, 60, 60, 90])])
        assert finder.balloon(rgb, .01).target is None  # Needs a current color hit.
        report = finder.balloon(rgb, .02)
        assert report.target.label == 'green_balloon'
        assert report.target.age_ms == pytest.approx(20)
        assert report.target.seen_ms == 0
        assert report.avoid is None and report.ready
        assert finder.balloon(np.zeros_like(rgb), .03).target is None
        worker.detection('balloon', .03, [('green_balloon', [30, 60, 60, 90])])
        assert finder.goal(rgb, .04, 'orange').target is None
        assert finder.tracker.tracks == []
        worker.detection('goal', .04, [('orange_square', [100, 80, 140, 120]),
                                      ('yellow_square', [200, 80, 250, 130])])
        finder.goal(rgb, .05, 'orange')
        report = finder.goal(rgb, .06, 'orange')
        assert report.target.label == 'orange_square'
        assert report.avoid.label == 'yellow_square'
        assert finder.goal(rgb, .07, 'yellow').target.label == 'yellow_square'
        assert finder.balloon(rgb, .08).avoid is None
        assert not finder.tracker.tracks
        worker.detection('balloon', 0, [('green_balloon', [30, 60, 60, 90])])
        finder.balloon(rgb, .09)
        assert not finder.tracker.tracks  # Old same-phase result from before the switch.
        assert worker.submissions[-1] == (.09, 'balloon')
        with pytest.raises(ValueError):
            finder.goal(rgb, .1, 'blue')
    assert worker.state == 'stopped'


def test_best_and_runtime_scaling():
    tracks = [Track(1, 'green_balloon', [0, 0, 10, 10], 1, 1),
              Track(2, 'green_balloon', [0, 0, 20, 10], 0, 0),
              Track(3, 'green_balloon', [0, 0, 20, 10], 2, 2)]
    assert best_track(tracks, ['green_balloon']).id == 3
    tracks[2].misses = 1
    assert best_track(tracks, ['green_balloon']).id == 2
    finder = FindTarget(worker=StubWorker(), lens=LensModel.ideal_equidistant(640, 480, axis='horizontal'))
    assert finder.lens.width == 320
    assert finder.lens.K[0, 0] == pytest.approx(LensModel.ideal_equidistant(axis='horizontal').K[0, 0])


def test_fake_interpolation_gaps_and_avoid():
    with FakeFindTarget([(0, (4, 0, 0)), (2, (2, 2, 2)), (3, None),
                         (4, (1, 0, 0))], [(0, (5, -2, 0)), (2, (3, -2, 0))]) as fake:
        report = fake.balloon(None, 1)
        assert report.target.body_xyz_m == (3, 1, 1)
        assert report.target.cam_xyz_m == (1, -1, 3)
        assert report.avoid is None and report.ready
        assert fake.goal(None, 1, 'orange').avoid.body_xyz_m == (4, -2, 0)
        for t in (-1, 2.5, 3, 3.5, 5):
            assert fake.balloon(None, t).target is None
        assert fake.balloon(None, 4).target is not None
    assert not fake.ready
    with pytest.raises(ValueError):
        FakeFindTarget([(1, (1, 0, 0)), (0, (2, 0, 0))])


def test_geometry_failure_drops_target_not_loop():
    class Failing:
        lens = LensModel.ideal_equidistant(axis='horizontal')

        def from_track(self, track, frame_time):
            raise ValueError('Fisheye inverse failed')

    worker = StubWorker()
    with FindTarget(worker=worker, geometry=Failing()) as finder:
        rgb = frame()
        finder.balloon(rgb, 0)
        worker.detection('balloon', 0, [('green_balloon', [30, 60, 60, 90])])
        finder.balloon(rgb, .01)
        report = finder.balloon(rgb, .02)
        assert report.target is None and report.ready
        assert finder.tracker.tracks and finder.geometry_failures >= 1
        assert worker.submissions[-1] == (.02, 'balloon')


def test_squoval_keeps_edges_and_drops_far_corners():
    lens = LensModel.ideal_equidistant()
    inside = lens.in_view([[160, 120], [2, 120], [160, 2], [60, 40]])
    assert inside.all()  # Center, edge mid-lines, and off-diagonal points stay.
    assert not lens.in_view([[5, 5], [315, 235]]).any()
    u, v = np.meshgrid(np.arange(320) + .5, np.arange(240) + .5)
    kept = lens.in_view(np.column_stack((u.ravel(), v.ravel()))).mean()
    assert kept == pytest.approx(.927, abs=.01)


def test_corner_track_is_not_reported():
    worker = StubWorker()
    with FindTarget(worker=worker) as finder:
        rgb = np.zeros((240, 320, 3), np.uint8)
        rgb[2:22, 2:22] = cv2.cvtColor(np.uint8([[[60, 255, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        finder.balloon(rgb, 0)
        worker.detection('balloon', 0, [('green_balloon', [2, 2, 22, 22])])
        finder.balloon(rgb, .01)
        report = finder.balloon(rgb, .02)
        assert finder.tracker.tracks and report.target is None


@pytest.mark.parametrize('color', ['orange', 'yellow'])
@pytest.mark.parametrize('shape', ['circle', 'square', 'triangle'])
def test_find_target_accepts_each_goal_shape_and_nearest_ranking(color, shape):
    from dtr.target_geometry import load_target_sizes
    sizes = load_target_sizes()
    label = f'{color}_{shape}'
    w, h = sizes[label]['width_m']/.03, sizes[label]['height_m']/.03
    nearest = Track(1, label, [160-w/2, 120-h/2, 160+w/2, 120+h/2], 1., 1.)
    farther = Track(2, f'{color}_triangle', [140, 100, 180, 135], 1., 1.)
    assert best_track([farther, nearest], [f'{color}_{s}' for s in ('circle','square','triangle')], sizes) is nearest
    worker = StubWorker()
    with FindTarget(worker=worker) as finder:
        rgb = frame()
        finder.goal(rgb, 0, color)
        # Seed current colour tracks directly to isolate routing from async search latency.
        finder.tracker.tracks = [nearest]
        nearest.box = [100., 80., 140., 120.] if color == 'orange' else [200., 80., 250., 130.]
        nearest.confirmed_wh = (40., 40.) if color == 'orange' else (50., 50.)
        report = finder.goal(rgb, 1., color)
        assert report.target.label == label and report.avoid is None
        report = finder.goal(rgb, 1.01, 'yellow' if color == 'orange' else 'orange')
        assert report.target is None and report.avoid.label == label
