#!/usr/bin/env python3
"""Exercise invented E2 samples offline, or explicitly record them on the bench relay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_balloon_hunt_demo import decode_sim, encode_sim, generate_records


BENCH = {"bench_only": True, "applied": False}


def fixture(session=None):
    records = generate_records() if session is None else generate_records(session=session)
    packets = [encode_sim(record) for record in records]
    for packet in packets:
        if encode_sim(decode_sim(packet)) != packet:
            raise ValueError("fixture roundtrip failed")
    return records, packets


def run(client=None, *, op=None, clock=time.monotonic, sleep=time.sleep,
        cancelled=lambda: False, log=None, interval_ms=50, ack_timeout_ms=50, paced=False):
    """No reconnect/replay. Late slots expire; every send uses its own slot.

    sent counts mission API calls, not UDP datagrams (BenchClient can retry the same
    identity). Unknown acceptance stops the run and remains explicitly unknown.
    """
    if type(interval_ms) is not int or interval_ms not in (50, 100, 200):
        raise ValueError("bench interval must be 50, 100 or 200 ms")
    if type(ack_timeout_ms) is not int or ack_timeout_ms not in (50, 100, 150, 200):
        raise ValueError("bench acknowledgement timeout must be 50, 100, 150 or 200 ms")
    interval = interval_ms / 1000
    records, packets = fixture(secrets.randbelow(0xFFFFFFFF) + 1 if client else None)
    start = clock()
    summary = dict(status="incomplete", scheduled=len(packets), sent=0, accepted=0,
                   skipped=0, rejected=0, unknown=0, validated=len(packets), **BENCH)
    summary.update(replay_interval_ms=interval_ms, ack_timeout_ms=ack_timeout_ms,
                   timing_mode="paced_recording" if paced else "scheduled_slots",
                   source_scenario_duration_ms=20000,
                   requested_replay_duration_ms=len(packets) * interval_ms)
    summary["fixture_sha256"] = hashlib.sha256(b"".join(packets)).hexdigest()
    summary["session"] = records[0]["session"]

    def emit(event):
        if log is not None:
            try:
                log.write(json.dumps({**event, **BENCH}, sort_keys=True) + "\n")
                log.flush()
            except (OSError, ValueError):
                summary.update(log_failed=True, reason="log_failed")
                if summary["status"] != "cancelled":
                    summary["status"] = "incomplete"
                return False
        return True

    if not emit({"event": "start", **summary}):
        summary["elapsed_seconds"] = round(clock() - start, 6)
        return summary
    try:
        if client is None:
            summary["status"] = "complete"
        else:
            result = client.connect()
            if result.get("bench_only") is not True or result.get("applied") is not False:
                raise ValueError("invalid bench handshake")
            # Cadence begins after the handshake, never catches up its delay.
            origin = clock()
            next_send = origin
            for record, packet in zip(records, packets):
                if cancelled():
                    summary["status"] = "cancelled"
                    break
                due = next_send if paced else origin + record["seq"] * interval
                ready = max(due, next_send)
                while True:
                    remaining = ready - clock()
                    if remaining <= 0:
                        break
                    sleep(min(0.01, remaining))
                    if cancelled():
                        break
                if cancelled():
                    summary["status"] = "cancelled"
                    break
                if not paced and clock() >= due + interval:
                    summary["skipped"] += 1
                    if not emit({"event": "skipped", "seq": record["seq"]}):
                        break
                    continue
                next_send = clock() + interval
                summary["sent"] += 1
                try:
                    result = client.mission(op, packet, timeout=ack_timeout_ms / 1000)
                except (TimeoutError, OSError, ValueError):
                    summary["unknown"] += 1
                    emit({"event": "acceptance_unknown", "seq": record["seq"]})
                    break
                if result.get("bench_only") is not True or result.get("applied") is not False:
                    summary["unknown"] += 1
                    break
                status = result.get("status")
                if status == "QUEUED_BENCH":
                    summary["accepted"] += 1
                elif status == "BUSY":
                    summary["rejected"] += 1
                else:
                    summary["unknown"] += 1
                if not emit({"event": "response", "seq": record["seq"],
                             "packet_sha256": hashlib.sha256(packet).hexdigest(), "status": status}):
                    break
                if status != "QUEUED_BENCH":
                    break
            if summary["accepted"] == len(packets) and not summary.get("log_failed"):
                summary["status"] = "complete"
    except KeyboardInterrupt:
        summary["status"] = "cancelled"
        # Interrupt during a mission cannot establish whether it was recorded.
        summary["unknown"] = summary["sent"] - summary["accepted"] - summary["rejected"]
    except (OSError, ValueError, TimeoutError):
        summary["reason"] = "bench_session_failed"
    summary["elapsed_seconds"] = round(clock() - start, 6)
    emit({"event": "summary", **summary})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transmit-bench", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--key-file")
    parser.add_argument("--op", type=int, help="opaque unassigned bench mapping, uint8")
    parser.add_argument("--port", type=int, default=47100)
    parser.add_argument("--log", help="new private JSONL recording; never overwritten")
    parser.add_argument("--interval-ms", type=int, choices=(50, 100, 200), default=50,
                        help="bench replay spacing; source fixture remains 20 seconds")
    parser.add_argument("--ack-timeout-ms", type=int, choices=(50, 100, 150, 200), default=50)
    parser.add_argument("--paced", action="store_true",
                        help="record every synthetic sample at a minimum spacing; not real-time replay")
    args = parser.parse_args(argv)
    if args.transmit_bench:
        if not args.host or not args.key_file or args.op is None:
            parser.error("--transmit-bench requires --host, --key-file and --op")
        if not 0 <= args.op <= 255 or not 1 <= args.port <= 65535:
            parser.error("op must be uint8 and port must be 1..65535")
    elif args.host or args.key_file or args.op is not None:
        parser.error("network options require --transmit-bench")
    log = None
    try:
        if args.log:
            fd = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            log = os.fdopen(fd, "w", encoding="ascii")
        if args.transmit_bench:
            from dtr.ground_link import BenchClient, load_key
            with BenchClient(load_key(args.key_file), args.host, args.port) as client:
                result = run(client, op=args.op, log=log, interval_ms=args.interval_ms,
                             ack_timeout_ms=args.ack_timeout_ms, paced=args.paced)
        else:
            result = run(log=log, interval_ms=args.interval_ms,
                         ack_timeout_ms=args.ack_timeout_ms, paced=args.paced)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "complete" else 130 if result["status"] == "cancelled" else 1
    except KeyboardInterrupt:
        print(json.dumps(dict(status="cancelled", **BENCH)))
        return 130
    except (OSError, ValueError):
        print(json.dumps(dict(status="incomplete", reason="setup_failed", **BENCH)))
        return 1
    finally:
        if log is not None:
            try:
                log.close()
            except OSError:
                pass  # Writes are flushed and any failure is already in the stdout summary.


if __name__ == "__main__":
    raise SystemExit(main())
