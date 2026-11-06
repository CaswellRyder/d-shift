"""Camera-free reference guidance cycle at 20 Hz."""
import argparse
import time
from dtr.flight import encode_i2c, decode_i2c
from dtr.esp32_link import I2CLink

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--i2c-bus', type=int, default=1)
    parser.add_argument('--i2c-address', type=lambda s: int(s, 0), default=0x42)
    args = parser.parse_args()
    link = I2CLink(args.i2c_bus, args.i2c_address)
    start = time.monotonic()
    seq = 0
    cycle = [('SEARCH', 0), ('TRACK', -20), ('TRACK', 20), ('FAN_ON', 0), ('LOST', 0), ('ENTER', 0)]
    try:
        while time.monotonic() - start < args.seconds:
            cmd, yaw = cycle[(seq // 20) % len(cycle)]
            packet = encode_i2c(dict(seq=seq, cmd=cmd, phase='goal' if cmd == 'ENTER' else 'balloon',
                goal_color='yellow', fan=cmd == 'FAN_ON', search_dir=0,
                target=None if cmd == 'SEARCH' else dict(label='green_balloon', yaw_deg=yaw,
                elev_m=0, travel_m=1, age_ms=100)))
            ok = link.send(packet)
            print(packet.hex(), decode_i2c(packet), 'ok' if ok else link.last_error)
            seq += 1
            time.sleep(max(0, start + seq / 20 - time.monotonic()))
    finally:
        link.close()

if __name__ == '__main__':
    main()
