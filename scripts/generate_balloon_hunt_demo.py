#!/usr/bin/env python3
"""Write an offline, invented balloon-hunt fixture; never transmit or actuate.

The 20 Hz cadence, geometry and freshness are examples, not model measurements.
E1/E2/E3 are a proposed 24-byte controller contract, separate from legacy D7 v2.
All generated examples default to SHADOW, with no actuator authorization.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import struct


DEFAULT_SESSION = 0x13572468
PACKET_SIZE = 24
VERSION = 1
SIM_SOURCE = 0x53
COMMANDS = ("SEARCH", "TRACK", "LOST", "FAN_ON", "ENTER", "AVOID")
PHASES = ("balloon", "capture", "goal")
LABELS = (
    "green_balloon", "purple_balloon", "orange_circle", "orange_square",
    "orange_triangle", "yellow_circle", "yellow_square", "yellow_triangle",
)
OPERATIONS = ("HEARTBEAT", "MODE_SHADOW", "MODE_SCRIPT", "SIM_START", "SIM_CANCEL", "MODE_OFF")
SIM_MODES = ("OFF", "SHADOW", "SCRIPT")
SIM_FORMAT = "<BBIHBBBBBBhhHHB"
OPERATOR_FORMAT = "<BBIHBBBBhhhhBBB"
GEOMETRY = ("yaw_deg", "elev_m", "travel_m")


def crc8(data):
    """CRC-8 polynomial 0x07, initial 0; no reflection or final xor."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ (0x07 if crc & 0x80 else 0)) & 0xFF
    return crc


def _integer(name, value, low, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer in the permitted range")
    return value


def _boolean(name, value):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _index(name, values, value):
    if type(value) is not str or value not in values:
        raise ValueError(f"Unknown {name}")
    return values.index(value)


def _scaled(name, value, low, high, missing):
    if value is None:
        return missing
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite or None")
    if not low / 100 <= value <= high / 100:
        raise ValueError(f"{name} is outside the wire range")
    return round(value * 100)


def _time(name, value, maximum, unit=1, missing=0xFFFF):
    if value is None:
        return missing
    return min(_integer(name, value, 0) // unit, maximum)


def _sim_fields(record):
    if record.get("source") != "SIM":
        raise ValueError("Only source SIM is permitted")
    if record.get("simulated") is not True or record.get("actuator_authorized") is not False:
        raise ValueError("SIM fixtures must be simulated and actuator unauthorized")
    session = _integer("session", record.get("session"), 1, 0xFFFFFFFF)
    seq = _integer("seq", record.get("seq"), 0, 0xFFFF)
    cmd = _index("command", COMMANDS, record.get("cmd"))
    phase = _index("phase", PHASES, record.get("phase"))
    label = record.get("label")
    label_index = 255 if label is None else _index("label", LABELS, label)
    geometry = [_scaled(key, record.get(key), 0 if key == "travel_m" else -32767,
                        65534 if key == "travel_m" else 32767,
                        0xFFFF if key == "travel_m" else -32768) for key in GEOMETRY]
    fan = _boolean("fan_requested", record.get("fan_requested"))
    follow = _boolean("follow_fresh", record.get("follow_fresh"))
    end = _boolean("end_of_script", record.get("end_of_script"))
    direction = _integer("search_dir", record.get("search_dir"), -1, 1)
    age = _time("age_ms", record.get("age_ms"), 0xFFFE)
    lost = _time("lost_ms", record.get("lost_ms"), 254, unit=100, missing=255)
    if direction and (cmd != 0 or end):
        raise ValueError("Search direction is valid only for active SEARCH")
    if cmd == 0 and (label is not None or any(record.get(key) is not None for key in GEOMETRY)
                     or record.get("age_ms") is not None or fan or follow):
        raise ValueError("SEARCH has no target geometry, age, fan request or follow freshness")
    if (cmd == 2) != (record.get("lost_ms") is not None) or (cmd == 2 and follow):
        raise ValueError("LOST requires loss time and cannot claim follow freshness")
    if fan and cmd != 3:
        raise ValueError("A fan request requires FAN_ON")
    if end and (cmd != 0 or direction):
        raise ValueError("END must use inert SEARCH without a search direction")
    flags = sum(1 << i for i, key in enumerate(GEOMETRY) if record.get(key) is not None)
    flags |= (int(fan) << 3) | (int(follow) << 4) | (int(end) << 5)
    flags |= (int(direction == 1) << 6) | (int(direction == -1) << 7)
    if "flags" in record and _integer("flags", record["flags"], 0, 255) != flags:
        raise ValueError("Flags disagree with semantic fields")
    return session, seq, cmd, phase, label_index, flags, *geometry, age, lost


def encode_sim(record):
    """Encode E2; range errors, nonfinite values and bool-as-number are rejected.

    Geometry rounds to nearest centi-unit. Known age saturates at 65534 ms;
    loss time floors to 100 ms units, saturating at 25400 ms. None is distinct
    from zero. Extra descriptive JSON fields do not enter the wire packet.
    """
    session, seq, cmd, phase, label, flags, yaw, elev, travel, age, lost = _sim_fields(record)
    body = struct.pack(SIM_FORMAT, 0xE2, VERSION, session, seq, SIM_SOURCE,
                       cmd, phase, label, flags, 0, yaw, elev, travel, age, lost)
    return body + bytes([crc8(body)])


def decode_sim(packet):
    """Validate an E2 packet and return its semantic fields in SI units.

    Decoding preserves the quantized/saturated wire values. END must be handled
    before cmd: it cancels SCRIPT and returns SHADOW, including repeated END.
    """
    if not isinstance(packet, (bytes, bytearray)) or len(packet) != PACKET_SIZE:
        raise ValueError("E2 must contain exactly 24 bytes")
    if packet[:2] != b"\xe2\x01" or crc8(packet[:-1]) != packet[-1]:
        raise ValueError("Invalid E2 magic, version or CRC")
    (_, _, session, seq, source, cmd, phase, label, flags, reserved,
     yaw, elev, travel, age, lost) = struct.unpack(SIM_FORMAT, packet[:-1])
    if source != SIM_SOURCE or reserved:
        raise ValueError("Invalid SIM source or reserved byte")
    if cmd >= len(COMMANDS) or phase >= len(PHASES) or (label != 255 and label >= len(LABELS)):
        raise ValueError("Unknown E2 enum")
    if flags & 0xC0 == 0xC0:
        raise ValueError("Both search directions cannot be set")
    values = (yaw, elev, travel)
    for i, (value, sentinel) in enumerate(zip(values, (-32768, -32768, 65535))):
        if bool(flags & (1 << i)) == (value == sentinel):
            raise ValueError("Geometry validity bit disagrees with sentinel")
    record = dict(
        session=session, seq=seq, source="SIM", simulated=True, actuator_authorized=False,
        cmd=COMMANDS[cmd], phase=PHASES[phase], label=None if label == 255 else LABELS[label],
        yaw_deg=None if yaw == -32768 else yaw / 100,
        elev_m=None if elev == -32768 else elev / 100,
        travel_m=None if travel == 65535 else travel / 100,
        age_ms=None if age == 65535 else age,
        lost_ms=None if lost == 255 else lost * 100,
        fan_requested=bool(flags & 8), follow_fresh=bool(flags & 16),
        end_of_script=bool(flags & 32),
        search_dir=1 if flags & 64 else -1 if flags & 128 else 0, flags=flags,
    )
    _sim_fields(record)
    return record


def encode_operator(*, session=DEFAULT_SESSION, seq=0, op="HEARTBEAT",
                    requested_sim_mode="SHADOW", deadman=False, manual_axes_valid=False,
                    ttl_10ms=30, forward=0.0, right=0.0, up=0.0, yaw=0.0, arg=0):
    """Encode proposed E1 normalized intent, never physical PWM or an I/O action."""
    session = _integer("session", session, 1, 0xFFFFFFFF)
    seq = _integer("seq", seq, 0, 0xFFFF)
    operation = _index("operator op", OPERATIONS, op)
    mode = _index("sim mode", SIM_MODES, requested_sim_mode)
    _boolean("deadman", deadman)
    _boolean("manual_axes_valid", manual_axes_valid)
    ttl_10ms = _integer("ttl_10ms", ttl_10ms, 1, 30)
    arg = _integer("arg", arg, 0, 255)
    requested = {"MODE_SHADOW": "SHADOW", "MODE_SCRIPT": "SCRIPT", "MODE_OFF": "OFF"}
    if op in requested and requested_sim_mode != requested[op]:
        raise ValueError("Mode operation and requested mode disagree")
    if (op == "SIM_START" and arg != 1) or (op != "SIM_START" and arg != 0):
        raise ValueError("SIM_START needs scenario arg 1; other ops need arg 0")
    axes = []
    for name, value in zip(("forward", "right", "up", "yaw"), (forward, right, up, yaw)):
        if type(value) not in (int, float) or not -1 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite normalized intent in [-1, 1]")
        if not manual_axes_valid and value != 0:
            raise ValueError("Invalid manual axes must carry zero intent")
        axes.append(round(value * 1000))
    flags = int(deadman) | (int(manual_axes_valid) << 1)
    body = struct.pack(OPERATOR_FORMAT, 0xE1, VERSION, session, seq,
                       operation, mode, flags, ttl_10ms, *axes, arg, 0, 0)
    return body + bytes([crc8(body)])


def _lerp(start, finish, index, count):
    return start + (finish - start) * index / (count - 1)


def generate_records(session=DEFAULT_SESSION):
    """Return exactly 400 invented samples covering [0, 20 s), at 50 ms steps."""
    _integer("session", session, 1, 0xFFFFFFFF)
    records = []
    for seq in range(400):
        t_ms = seq * 50
        record = dict(session=session, seq=seq, t_ms=t_ms, source="SIM", simulated=True,
                      actuator_authorized=False, step="SEARCH_RIGHT", cmd="SEARCH",
                      phase="balloon", label=None, yaw_deg=None, elev_m=None, travel_m=None,
                      age_ms=None, lost_ms=None, fan_requested=False, follow_fresh=False,
                      search_dir=1, end_of_script=False)
        if 3000 <= t_ms < 9000 or 12000 <= t_ms < 18000:
            record.update(cmd="TRACK", label="green_balloon", search_dir=0,
                          age_ms=(seq % 8) * 50)
            if t_ms < 4000:
                step, geometry = "ACQUIRE", (25.0, 0.5, 6.0)
            elif t_ms < 6000:
                n = (t_ms - 4000) // 50
                step = "TURN_TO_TARGET"
                geometry = tuple(_lerp(a, b, n, 40) for a, b in
                                 zip((25.0, 0.5, 6.0), (5.0, 0.2, 4.5)))
            elif t_ms < 9000:
                n = (t_ms - 6000) // 50
                step = "ALIGN_APPROACH"
                geometry = tuple(_lerp(a, b, n, 60) for a, b in
                                 zip((5.0, 0.2, 4.5), (-4.0, 0.0, 2.0)))
            elif t_ms < 13000:
                step, geometry = "REACQUIRE", (-10.0, 0.0, 2.2)
            elif t_ms < 15000:
                n = (t_ms - 13000) // 50
                step = "APPROACH"
                geometry = (_lerp(-10.0, 0.0, n, 40), 0.0, _lerp(2.2, 1.0, n, 40))
            else:
                step = "FAN_REQUEST"
                geometry = (0.0, 0.0, _lerp(1.0, 0.8, (t_ms - 15000) // 50, 60))
                record.update(cmd="FAN_ON", phase="capture", fan_requested=True)
            record.update(step=step, **dict(zip(GEOMETRY, geometry)))
        elif 9000 <= t_ms < 11000:
            previous = records[-1]
            record.update(step="LOST", cmd="LOST", label="green_balloon", search_dir=0,
                          age_ms=previous["age_ms"] + 50, lost_ms=t_ms - 8950,
                          **{key: previous[key] for key in GEOMETRY})
        elif 11000 <= t_ms < 12000:
            record.update(step="SEARCH_LEFT", search_dir=-1)
        elif t_ms >= 18000:
            record.update(step="END", search_dir=0, end_of_script=True)
        record["packet_hex"] = encode_sim(record).hex()
        records.append(record)
    return records


def operator_examples(session):
    """Inert examples; none requests SCRIPT, enables axes, or asserts deadman."""
    examples = []
    for seq, (name, op, arg) in enumerate((
        ("MODE_SHADOW", "MODE_SHADOW", 0), ("SIM_START", "SIM_START", 1),
        ("SIM_CANCEL", "SIM_CANCEL", 0), ("MANUAL_ZERO_INVALID", "HEARTBEAT", 0),
    )):
        example = dict(name=name, session=session, seq=seq, op=op, requested_sim_mode="SHADOW",
                       deadman=False, manual_axes_valid=False, ttl_10ms=30,
                       forward=0.0, right=0.0, up=0.0, yaw=0.0, arg=arg)
        packet = encode_operator(**{key: value for key, value in example.items() if key != "name"})
        example.update(expected_owner="PILOT", actuator_authorized=False, packet_hex=packet.hex())
        examples.append(example)
    return examples


def write_demo(output, session=DEFAULT_SESSION):
    """Create an exclusive NEW_DIR and offline artifacts; existing paths fail."""
    records = generate_records(session)
    examples = operator_examples(session)
    summary = dict(
        scenario="hunt-balloon-v1", session=session, duration_ms=20000, sample_period_ms=50,
        record_count=len(records), packet_bytes=PACKET_SIZE, binary_bytes=len(records) * PACKET_SIZE,
        default_sim_mode="SHADOW", default_owner="PILOT", simulated=True,
        actuator_authorized=False, actual_forwarding=False, measured_geometry=False,
        measured_inference_cadence=False, verified_capture=False,
        command_counts=dict(Counter(record["cmd"] for record in records)),
        step_counts=dict(Counter(record["step"] for record in records)),
        final_action="END cancels SCRIPT and returns SHADOW; process END before cmd",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "hunt-balloon-v1.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    with (output / "hunt-balloon-v1.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "hunt-balloon-v1.bin").open("xb") as stream:
        for record in records:
            stream.write(bytes.fromhex(record["packet_hex"]))
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "operator-examples.jsonl").open("x", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, allow_nan=False) + "\n")
    with (output / "operator-examples.txt").open("x", encoding="utf-8") as stream:
        stream.write("OFFLINE PROPOSED E1 EXAMPLES: no transmission or actuator authorization.\n"
                     "Default owner PILOT; mode SHADOW; TTL proposal 300 ms.\n"
                     "All axes zero AND invalid; deadman false. MANUAL uses HEARTBEAT.\n"
                     "SIM_START scenario 1 is a shadow preview; it does not enable SCRIPT.\n")
        for example in examples:
            stream.write(f"{example['name']}: {example['packet_hex']}\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory; never overwritten")
    parser.add_argument("--session", type=lambda value: int(value, 0), default=DEFAULT_SESSION,
                        help="Nonzero u32 session, decimal or 0x hex (default: 0x13572468)")
    args = parser.parse_args(argv)
    try:
        summary = write_demo(args.output, args.session)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
