import pytest
from dtr.flight import COMMANDS, LABELS, PHASES, encode_i2c, decode_i2c, Mission, Track
from dtr.esp32_link import I2CLink

def test_packet():
    for cmd in COMMANDS:
        for phase in PHASES:
            for label in LABELS:
                p = encode_i2c(dict(cmd=cmd, phase=phase, target=dict(label=label)))
                assert len(p) == 20
                d = decode_i2c(p)
                assert d['cmd'] == cmd and d['phase'] == phase
                assert d['label'] == (None if cmd == 'SEARCH' else label)
    p = bytearray(p)
    p[8] ^= 1
    with pytest.raises(ValueError):
        decode_i2c(p)

def test_saturation():
    d = decode_i2c(encode_i2c(dict(cmd='TRACK', seq=10**99,
        target=dict(yaw_deg=float('inf'), elev_m=-1e99, travel_m=1e99, age_ms=1e99))))
    assert (d['seq'], d['yaw_deg'], d['elev_m'], d['travel_m'], d['age_ms']) == (65535,327.67,-327.68,655.34,65534)

def test_link():
    now = [0]
    opens = []
    def opener(*args):
        opens.append(args)
        return 10
    def bad(*args):
        raise OSError('NACK')
    link = I2CLink(opener=opener, ioctl=lambda *a: None, writer=lambda f,p: len(p),
                   closer=lambda f: None, clock=lambda: now[0])
    assert link.send(b'abc') and link.sent == 1
    link.writer = bad
    assert not link.send(b'a') and link.errors == 1
    # A NACK is per transfer: no reopen, no back-off, the next frame retries.
    assert not link.send(b'a') and link.errors == 2 and len(opens) == 1
    link.writer = lambda f, p: len(p)
    assert link.send(b'a') and link.sent == 2 and len(opens) == 1


def test_link_open_failure_backs_off():
    now = [0]
    opens = []
    def opener(*args):
        opens.append(args)
        raise OSError('No such file')
    link = I2CLink(opener=opener, ioctl=lambda *a: None, writer=lambda f, p: len(p),
                   closer=lambda f: None, clock=lambda: now[0], retry_s=1.0)
    assert not link.send(b'a') and len(opens) == 1 and link.errors == 1
    assert not link.send(b'a') and len(opens) == 1  # Rate-limited, not counted again.
    now[0] = 1
    assert not link.send(b'a') and len(opens) == 2
    link.opener = lambda *a: 10
    now[0] = 2
    assert link.send(b'a') and link.sent == 1

def test_follow_freshness():
    mission = Mission('yellow')
    track = Track(1, 'green_balloon', [100,80,220,160], confirmed=1, seen=2.5,
                  confirmed_wh=(120,80), follow_start=0)
    target = mission.describe(track, 2.5)
    assert mission._ready(target, track, .3)
    track.last_miss = 1.2
    assert not mission._ready(target, track, .3)
    # Applying a late confirmation leaves the intervening miss intact.
    track.confirmed = 1.1
    assert not mission._ready(target, track, .3)
    track.last_miss = float('-inf')
    assert not mission.freshness(dict(age_ms=2501), track)

def test_late_confirmation_preserves_miss():
    import numpy as np
    from dtr.flight import ColorTracker
    tracker = ColorTracker()
    tracker.merge([dict(label='green_balloon', box=[100,80,220,160])], 0, 0)
    track = tracker.tracks[0]
    tracker.update(np.zeros((240,320,3), dtype=np.uint8), 1.2)
    tracker.merge([dict(label='green_balloon', box=[100,80,220,160])], 1.0, 1.5)
    assert track.last_miss == 1.2
    track.misses = 0
    assert not Mission('yellow').freshness(dict(age_ms=1500), track)


def test_jump_to_other_same_colour_blob_breaks_follow():
    import numpy as np
    from dtr.flight import ColorTracker
    green = (40, 180, 40)
    def image(cx):
        rgb = np.zeros((240, 320, 3), np.uint8)
        import cv2
        cv2.circle(rgb, (cx, 120), 20, green, -1)
        return rgb
    tracker = ColorTracker()
    tracker.merge([dict(label='green_balloon', box=[140, 100, 180, 140])], 0.0, 0.0)
    track = tracker.tracks[0]
    tracker.update(image(160), 0.05)
    tracker.merge([dict(label='green_balloon', box=[140, 100, 180, 140])], 0.05, 0.6)
    tracker.update(image(160), 0.65)
    mission = Mission('yellow')
    assert track.misses == 0 and mission.freshness(dict(age_ms=1500), track)
    # The balloon vanishes and another green object of the same size sits 30 px away.
    tracker.update(image(190), 0.7)
    assert track.misses == 0 and track.last_miss == 0.7
    assert not mission.freshness(dict(age_ms=1500), track)


def test_frozen_lost_target_is_not_followed_and_ages():
    mission = Mission('yellow')
    track = Track(1, 'green_balloon', [140, 100, 180, 140], confirmed=1.0, seen=2.0,
                  confirmed_wh=(40, 40), follow_start=0)
    seen = mission.step([track], 2.0)
    assert seen['cmd'] == 'TRACK' and seen['target']['follow']
    lost = mission.step([], 2.5)
    assert lost['cmd'] == 'LOST' and lost['target']['follow'] is False
    assert lost['target']['age_ms'] == seen['target']['age_ms'] + 500
    assert not decode_i2c(encode_i2c(lost))['follow']
