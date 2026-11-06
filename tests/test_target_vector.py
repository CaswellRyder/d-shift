import numpy as np
import pytest
from dtr.lens import LensModel
from dtr.target_geometry import load_target_sizes
from dtr.target_vector import VectorCreator

@pytest.mark.parametrize('dist', [[0,0,0,0], [.01,-.002,.0003,-.00001]])
def test_inverse_and_synthetic_recovery(dist):
    lens = LensModel.ideal_equidistant()
    lens.D[:] = np.array(dist).reshape(4,1)
    v = VectorCreator(lens, load_target_sizes())
    for u,w in [(160,120),(60,70),(260,190),(109,147)]:
        x,y = v.pixel_to_flat(u,w)
        ray = np.array([x,y,1]); ray /= np.linalg.norm(ray)
        assert ray == pytest.approx(lens.rays([[u,w]])[0], abs=1e-9)
        d,H = 5, v.TARGET_HEIGHT_M['balloon']
        h = v.flat_to_pixel(x,y+H/2/d)[1]-v.flat_to_pixel(x,y-H/2/d)[1]
        yaw,elev,travel = v.target_vector(u,w,'balloon',h)
        assert travel == pytest.approx(np.hypot(x*d,d), rel=.01)
        assert elev == pytest.approx(-y*d, abs=.001)
    assert all(np.isfinite(v.target_vector(109,147,'triangle',47)))

def test_triangle_orientation_and_invalid_height():
    sizes = load_target_sizes(); lens = LensModel.ideal_equidistant()
    down = VectorCreator(lens,sizes); up = VectorCreator(lens,sizes,triangle_point_down=False)
    assert down.target_vector(160,120,'triangle',47)[1] > 0
    assert up.target_vector(160,120,'triangle',47)[1] < 0
    assert down.vector_for('yellow_triangle',[140,120,180,120]) is None

@pytest.mark.parametrize('x,sign', [(80,-1),(240,1),(160,0)])
def test_lost_frozen_search_and_reacquisition(x,sign):
    from dtr.flight import Mission, Track
    m=Mission('yellow')
    t=Track(1,'green_balloon',[x-5,115,x+5,125],0,0,confirmed_wh=(10,10))
    first=m.step([t],0)
    lost=m.step([],0.1)
    frozen = dict(first['target'], follow=False, age_ms=first['target']['age_ms']+100,
                  seen_ms=first['target']['seen_ms']+100)
    assert lost['cmd']=='LOST' and lost['target']==frozen
    assert lost['lost_ms']==100 and lost['search_dir']==sign
    assert m.step([],2)['cmd']=='SEARCH'
    assert m.step([],3)['search_dir']==sign
    assert m.step([t],3.1)['search_dir']==0
    m.captured()
    assert m.step([],3.2)['search_dir']==0


def test_enter_commit_and_scored():
    from dtr.flight import Mission, Track
    m=Mission('yellow');m.captured()
    t=Track(1,'yellow_square',[60,20,260,220],0,0,confirmed_wh=(200,200))
    assert m.step([t],0)['cmd']=='ENTER'
    assert m.step([],2.9)['cmd']=='ENTER'
    assert m.step([],3)['cmd']=='SEARCH'
    m.scored()
    assert m.step([],3.1)['cmd']=='SEARCH'

@pytest.mark.parametrize('shape', ['triangle','ring','L'])
def test_component_center_height(shape):
    import cv2
    from dtr.flight import ColorTracker
    hsv=np.zeros((100,100,3),np.uint8)
    mask=np.zeros((100,100),np.uint8)
    if shape=='triangle':
        cv2.fillPoly(mask,[np.array([[20,20],[80,20],[50,80]])],255)
    elif shape=='ring':
        cv2.circle(mask,(50,50),30,255,3)
    else:
        mask[20:81,20:25]=255;mask[70:81,20:81]=255
    hsv[mask>0]=(60,255,255)
    tracker=ColorTracker()
    box=tracker._hit(hsv,'green_balloon',(0,0,100,100),(50,50),65,2)
    assert box is not None
    if shape=='L': assert tracker._center_h==11
    else: assert tracker._center_h==box[3]-box[1]


def test_enter_stays_committed_while_the_close_hoop_fails_the_gate():
    from dtr.flight import Mission, Track
    m = Mission('yellow'); m.captured()
    t = Track(1, 'yellow_square', [60, 20, 260, 220], 0, 0, confirmed_wh=(200, 200))
    assert m.step([t], 0)['cmd'] == 'ENTER'
    # Hoop clips and swells past max_growth: the gate fails but the pass continues.
    t.box, t.seen = [-40, -60, 360, 300], 1.5
    assert m.step([t], 1.5)['cmd'] == 'ENTER'
    lost = m.step([], 2.0)
    assert lost['cmd'] == 'ENTER' and lost['lost_ms'] == 500
    assert m.step([], 3.1)['cmd'] == 'LOST'  # Commit window counts from the last gated ENTER.
    m.scored()
    assert m.step([], 3.2)['cmd'] == 'SEARCH'


def test_v2_line_fits_an_esp32_serial_buffer_in_the_worst_case():
    import json
    from dtr.flight import encode_v2
    message = dict(seq=10**9, t_ms=10**10, cmd='LOST', phase='capture', goal_color='yellow',
                   fan=False, lost_ms=99999, search_dir=-1,
                   target=dict(label='yellow_triangle', yaw_deg=-179.9, elev_m=-999.99,
                               travel_m=9999.99, age_ms=99999))
    line = encode_v2(message)
    assert line.endswith('\n') and len(line.encode()) <= 255  # Arduino-ESP32 default RX buffer is 256.
    assert list(json.loads(line)) == ['v', 'seq', 't_ms', 'cmd', 'phase', 'goal_color', 'fan', 'label',
                                      'yaw_deg', 'elev_m', 'travel_m', 'age_ms', 'lost_ms', 'search_dir']
    search = json.loads(encode_v2(dict(message, cmd='SEARCH')))
    assert search['label'] is search['yaw_deg'] is search['travel_m'] is None


def test_flight_loop_vector_uses_the_telemetry_lens():
    from types import SimpleNamespace
    from dtr.flight_telemetry import mapping_from_args
    from dtr.target_geometry import CONFIGS
    telemetry = mapping_from_args(SimpleNamespace(lens_calibration=None,
        camera_mount=str(CONFIGS / 'camera_mount.json'), target_sizes=str(CONFIGS / 'target_sizes.json')))
    v = VectorCreator(telemetry.lens, load_target_sizes())
    assert (v.FX, v.CX, v.CY) == (telemetry.lens.K[0, 0], telemetry.lens.K[0, 2], telemetry.lens.K[1, 2])


def test_partial_centre_column_falls_back_to_box_height():
    from dtr.flight import Mission, Track
    m = Mission('yellow'); m.captured()
    box = [140, 100, 180, 140]
    full = Track(1, 'yellow_triangle', box, 0, 0, confirmed_wh=(40, 40), center_h=36.)
    sliver = Track(1, 'yellow_triangle', box, 0, 0, confirmed_wh=(40, 40), center_h=4.)
    plain = Track(1, 'yellow_triangle', box, 0, 0, confirmed_wh=(40, 40))
    travel = lambda t: m.describe(t, 0)['travel_m']
    assert travel(sliver) == travel(plain)  # A 4 px top bar is not a 10x farther hoop.
    assert travel(full) > travel(plain)  # A believable centre line is still used.


def test_enter_commitment_does_not_transfer_to_another_hoop():
    from dtr.flight import Mission, Track
    m = Mission('yellow'); m.captured()
    first = Track(1, 'yellow_square', [60, 20, 260, 220], 0, 0, confirmed_wh=(200, 200))
    entered = m.step([first], 0)
    assert entered['cmd'] == 'ENTER'
    other = Track(2, 'yellow_square', [240, 80, 280, 120], .5, .5, confirmed_wh=(40, 40))
    held = m.step([other], .5)
    # Still the frozen pass through hoop 1, never an ENTER toward hoop 2.
    aged = dict(entered['target'], follow=False, age_ms=entered['target']['age_ms'] + 500,
                seen_ms=entered['target']['seen_ms'] + 500)
    assert held['cmd'] == 'ENTER' and held['target'] == aged and held['lost_ms'] == 500
    assert m.selected_track_id is None
    # The committed hoop reappearing keeps the commitment, even when it fails the gate.
    first.box = [-40, -60, 360, 300]
    assert m.step([first, other], 1.0)['target']['label'] == 'yellow_square'
    assert m.selected_track_id == 1
    # After the window the other hoop is tracked normally, not entered.
    assert m.step([other], 3.5)['cmd'] == 'TRACK' and m.selected_track_id == 2


def test_committed_pass_is_not_interrupted_by_an_opponent_hoop():
    from dtr.flight import Mission, Track
    m = Mission('yellow'); m.captured()
    assert m.step([Track(1, 'yellow_square', [60, 20, 260, 220], 0, 0, confirmed_wh=(200, 200))], 0)['cmd'] == 'ENTER'
    opponent = Track(3, 'orange_circle', [10, 10, 50, 50], .5, .5, confirmed_wh=(40, 40))
    assert m.step([opponent], .5)['cmd'] == 'ENTER'
    assert m.step([opponent], 3.5)['cmd'] == 'AVOID'


def test_balloon_depth_uses_width_when_highlight_shortens_height():
    from dtr.flight import Mission, Track
    m = Mission('yellow')
    full = Track(1, 'green_balloon', [150, 110, 170, 132.5], 0, 0, confirmed_wh=(20, 22.5))
    # The top of the balloon is a white highlight: the mask and centre line are short.
    clipped = Track(2, 'green_balloon', [150, 118, 170, 132.5], 0, 0, confirmed_wh=(20, 14.5),
                    center_h=14.5)
    near, short = m.describe(full, 0)['travel_m'], m.describe(clipped, 0)['travel_m']
    assert short == pytest.approx(near, rel=0.05)
