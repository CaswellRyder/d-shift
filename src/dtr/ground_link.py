"""Authenticated UDP bench relay. No flight, I2C, or actuator imports or writes.

Timings are proposed bench defaults, not flight-qualified limits. The sink seam is
``Receiver.drain_events``: mission events are queued, manual events replace the
latest slot. Acknowledgements report queue acceptance only, never application.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import hmac
import ipaddress
import json
import math
import os
import secrets
import socket
import stat
import threading
import time
from typing import Callable

PORT = 47100
MAX_DATAGRAM = 1024
FRESH_MS = 300
MANUAL_TIMEOUT_MS = 300
SESSION_TIMEOUT_MS = 2000
PENDING_TIMEOUT_MS = 1000
MISSION_CAPACITY = 16
MAX_PENDING = 4
MAX_INT = (1 << 63) - 1
FIELDS = {"v", "type", "session", "seq", "echo_tick", "payload", "tag"}
REQUESTS = {"hello", "ping", "manual", "mission"}
RESPONSES = {"welcome", "pong", "manual_ack", "ack"}
BENCH = {"applied": False, "bench_only": True}
Address = tuple[str, int] | tuple[str, int, int, int]


class ProtocolError(ValueError):
    """An invalid or unauthenticated packet; never include packet/key contents."""


def literal_endpoint(host: str, port: int) -> tuple[int, Address]:
    """Parse only numeric unicast literals; zones name interfaces on this machine.

    Avoid getaddrinfo/DNS and preserve the IPv6 scope in the kernel endpoint.
    Link-local destinations and binds require an explicit local interface zone.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    literal, separator, zone = host.partition("%")
    address = ipaddress.ip_address(literal)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("address must be explicit unicast")
    if address.version == 4:
        if separator or address == ipaddress.IPv4Address("255.255.255.255"):
            raise ValueError("invalid IPv4 unicast address")
        return socket.AF_INET, (str(address), port)
    scope = 0
    if separator:
        if not zone or "%" in zone:
            raise ValueError("invalid IPv6 interface zone")
        scope = int(zone) if zone.isascii() and zone.isdecimal() else socket.if_nametoindex(zone)
        if not 1 <= scope <= 0xffffffff:
            raise ValueError("invalid IPv6 interface scope")
    if address.is_link_local and not scope:
        raise ValueError("link-local IPv6 requires a local interface zone")
    return socket.AF_INET6, (str(address), port, 0, scope)


def source_endpoint(source: Address) -> Address:
    """Canonical source identity, including UDP port and IPv6 flow/scope fields."""
    if len(source) == 2:
        family, endpoint = literal_endpoint(*source)
        if family != socket.AF_INET:
            raise ValueError("IPv6 source requires a four-field endpoint")
        return endpoint
    if len(source) != 4:
        raise ValueError("invalid source endpoint")
    host, port, flow, scope = source
    if (type(flow) is not int or not 0 <= flow <= 0xfffff
            or type(scope) is not int or not 0 <= scope <= 0xffffffff):
        raise ValueError("invalid IPv6 source endpoint")
    family, endpoint = literal_endpoint(host if "%" in host or not scope else f"{host}%{scope}", port)
    if family != socket.AF_INET6:
        raise ValueError("invalid IPv6 source address")
    if endpoint[3] and endpoint[3] != scope:
        raise ValueError("conflicting IPv6 source scope")
    return endpoint[0], port, flow, scope


def _integer(value: object, minimum: int = 0, maximum: int = MAX_INT) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _hex(value: object, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(c in "0123456789abcdef" for c in value))


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate JSON key")
        result[key] = value
    return result


def encode(key: bytes, kind: str, session: str, seq: int, tick: int, payload: dict) -> bytes:
    envelope = {"v": 1, "type": kind, "session": session, "seq": seq,
                "echo_tick": tick, "payload": payload}
    envelope["tag"] = hmac.new(key, canonical(envelope), hashlib.sha256).hexdigest()
    wire = canonical(envelope)
    if len(wire) > MAX_DATAGRAM:
        raise ProtocolError("datagram too large")
    return wire


def decode(key: bytes, wire: bytes) -> dict:
    if not 0 < len(wire) <= MAX_DATAGRAM:
        raise ProtocolError("invalid datagram length")
    try:
        packet = json.loads(wire.decode("ascii"), object_pairs_hook=_unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                ProtocolError("non-finite JSON number")))
        if not isinstance(packet, dict) or set(packet) != FIELDS:
            raise ProtocolError("invalid envelope")
        if (type(packet["v"]) is not int or packet["v"] != 1
                or packet["type"] not in REQUESTS | RESPONSES
                or not isinstance(packet["session"], str)
                or (packet["session"] != "" and not _hex(packet["session"], 32))
                or not _integer(packet["seq"]) or not _integer(packet["echo_tick"])
                or not isinstance(packet["payload"], dict) or not _hex(packet["tag"], 64)):
            raise ProtocolError("invalid envelope fields")
        if canonical(packet) != wire:
            raise ProtocolError("non-canonical JSON")
        unsigned = {k: v for k, v in packet.items() if k != "tag"}
        expected = hmac.new(key, canonical(unsigned), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, packet["tag"]):
            raise ProtocolError("authentication failed")
        return packet
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError,
            RecursionError) as exc:
        raise ProtocolError("invalid authenticated datagram") from exc


def _payload(kind: str, payload: dict) -> bool:
    if kind == "hello":
        return set(payload) == {"nonce"} and _hex(payload["nonce"], 32)
    if kind == "ping":
        return not payload
    if kind == "mission":
        return (set(payload) == {"op", "args"} and _integer(payload["op"], 0, 255)
                and isinstance(payload["args"], str) and len(payload["args"]) <= 96
                and len(payload["args"]) % 2 == 0
                and _hex(payload["args"], len(payload["args"])))
    if kind == "manual":
        return (set(payload) == {"forward", "right", "up", "yaw", "hold"}
                and type(payload["hold"]) is bool
                and all(type(payload[k]) in (int, float) and -1 <= payload[k] <= 1
                        and math.isfinite(payload[k]) for k in ("forward", "right", "up", "yaw")))
    return False


def payload_digest(payload: dict) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_key(path: str | os.PathLike) -> bytes:
    """Read exactly 32 raw bytes from a private regular file, without symlinks."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("key must be a private regular file (0600 or stricter)")
        key = os.read(fd, 33)
        if len(key) != 32:
            raise ValueError("key must contain exactly 32 raw bytes")
        return key
    finally:
        os.close(fd)


def generate_key(path: str | os.PathLike) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(secrets.token_bytes(32))


@dataclass
class _Session:
    token: str
    source: Address
    created: int
    nonce: str
    last_seen: int
    sequences: dict[str, int] = field(default_factory=lambda: dict(ping=0, manual=0, mission=0))
    mission_payload: dict | None = None
    mission_status: str | None = None
    held: bool = False
    manual_tick: int = 0


class Receiver:
    """In-memory state machine; handle/poll do not call any sink or network API.

    ``clock`` returns monotonic seconds. Call ``poll`` at least every 50ms in the
    transport loop. ``drain_events`` is safe for a separate, potentially slow sink
    thread. Mission queue capacity is 16, plus one replaceable manual slot.
    """

    def __init__(self, key: bytes, *, clock: Callable[[], float] = time.monotonic,
                 token_factory: Callable[[], str] = lambda: secrets.token_hex(16)):
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        self.key, self.clock, self.token_factory = key, clock, token_factory
        self.active: _Session | None = None
        self.pending: dict[str, _Session] = {}
        self.hellos: deque[int] = deque()
        self.missions: deque[dict] = deque()
        self.latest_manual: dict | None = None
        self._lock = threading.Lock()

    def _now(self) -> int:
        return int(self.clock() * 1000)

    def _neutral(self, session: _Session, now: int, reason: str) -> None:
        self.latest_manual = {"kind": "manual", "session": session.token,
                              "seq": session.sequences["manual"], "receiver_tick": now,
                              "payload": dict(forward=0, right=0, up=0, yaw=0, hold=False),
                              "reason": reason, **BENCH}
        session.held = False

    def _poll(self, now: int) -> None:
        self.pending = {k: s for k, s in self.pending.items()
                        if now - s.created < PENDING_TIMEOUT_MS}
        while self.hellos and now - self.hellos[0] >= 1000:
            self.hellos.popleft()
        if self.active and now - self.active.last_seen >= SESSION_TIMEOUT_MS:
            if self.active.held:
                self._neutral(self.active, now, "session_expired")
            self.active = None
        if (self.active and self.active.held
                and now - self.active.manual_tick >= MANUAL_TIMEOUT_MS):
            self._neutral(self.active, now, "link_lost")

    def poll(self) -> None:
        with self._lock:
            self._poll(self._now())

    def close(self) -> None:
        with self._lock:
            if self.active and self.active.held:
                self._neutral(self.active, self._now(), "receiver_stopped")
            self.active = None
            self.pending.clear()

    def drain_events(self) -> list[dict]:
        with self._lock:
            events = list(self.missions)
            self.missions.clear()
            if self.latest_manual is not None:
                events.append(self.latest_manual)
                self.latest_manual = None
            return events

    def _reply(self, kind: str, session: _Session, seq: int, now: int, payload: dict) -> bytes:
        return encode(self.key, kind, session.token, seq, now, {**payload, **BENCH})

    def handle(self, wire: bytes, source: Address) -> bytes | None:
        try:
            source = source_endpoint(source)
        except (ValueError, OSError):
            return None
        with self._lock:
            now = self._now()
            self._poll(now)
            try:
                packet = decode(self.key, wire)
            except ProtocolError:
                return None
            kind, payload = packet["type"], packet["payload"]
            if not _payload(kind, payload):
                return None
            if kind == "hello":
                if packet["session"] or packet["seq"] != 0 or packet["echo_tick"] != 0:
                    return None
                if len(self.hellos) >= 5:
                    return None
                self.hellos.append(now)
                session = next((s for s in self.pending.values()
                                if s.source == source and s.nonce == payload["nonce"]), None)
                if session is None:
                    if len(self.pending) >= MAX_PENDING:
                        return None
                    token = self.token_factory()
                    if not _hex(token, 32) or token in self.pending or (
                            self.active and token == self.active.token):
                        return None
                    session = _Session(token, source, now, payload["nonce"], now)
                    self.pending[token] = session
                return self._reply("welcome", session, 0, now, {"nonce": session.nonce})
            session = self.active
            activating = session is None
            if activating:
                session = self.pending.get(packet["session"])
            if session is None or session.source != source or session.token != packet["session"]:
                return None
            seq = packet["seq"]
            last = session.sequences[kind]
            if kind == "mission" and last and seq == last:
                if payload == session.mission_payload:
                    return self._reply("ack", session, seq, now,
                                       {"status": session.mission_status,
                                        "digest": payload_digest(payload)})
                return None
            if seq <= last or (kind == "mission" and seq != last + 1):
                return None
            if not 0 <= now - packet["echo_tick"] <= FRESH_MS:
                return None
            if kind == "mission" and len(self.missions) >= MISSION_CAPACITY:
                # Rejection is a terminal decision for this identity. A delayed
                # duplicate must not become accepted after the queue drains.
                session.sequences[kind] = seq
                session.mission_payload, session.mission_status = payload, "BUSY"
                return self._reply("ack", session, seq, now,
                                   {"status": "BUSY", "digest": payload_digest(payload)})
            if activating:
                self.active = session
                self.pending.clear()
            session.sequences[kind] = seq
            session.last_seen = now
            if kind == "ping":
                return self._reply("pong", session, seq, now,
                                   {"state": {"manual_held": session.held,
                                              "mission_queued": len(self.missions)}})
            event = {"kind": kind, "session": session.token, "seq": seq,
                     "receiver_tick": now, "payload": payload, **BENCH}
            if kind == "manual":
                session.held = payload["hold"]
                session.manual_tick = now
                if not session.held:
                    event["payload"] = dict(forward=0, right=0, up=0, yaw=0, hold=False)
                    event["reason"] = "released"
                self.latest_manual = event
                return self._reply("manual_ack", session, seq, now,
                                   {"status": "QUEUED_BENCH", "digest": payload_digest(payload)})
            session.mission_payload = payload
            session.mission_status = "QUEUED_BENCH"
            self.missions.append(event)
            return self._reply("ack", session, seq, now,
                               {"status": "QUEUED_BENCH", "digest": payload_digest(payload)})


class ClientProtocol:
    """Pure client packet validation; no automatic mission replay after reconnect."""

    def __init__(self, key: bytes, *, clock: Callable[[], float] = time.monotonic):
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        self.key, self.clock = key, clock
        self.nonce = secrets.token_hex(16)
        self.session = ""
        self.tick = 0
        self.tick_received = 0.0
        self.sequences = dict(ping=0, manual=0, mission=0)
        # At most one locally timed request per channel. Retries retain the
        # original origin; they cannot make an old response look fresh.
        self.outstanding: dict[str, tuple[int, float, str]] = {}

    def hello(self) -> bytes:
        return encode(self.key, "hello", "", 0, 0, {"nonce": self.nonce})

    def welcome(self, wire: bytes) -> None:
        p = decode(self.key, wire)
        if (p["type"] != "welcome" or not p["session"] or p["seq"] != 0
                or p["payload"] != {"nonce": self.nonce, **BENCH}
                or p["payload"].get("applied") is not False
                or p["payload"].get("bench_only") is not True):
            raise ProtocolError("unexpected welcome")
        self.session, self.tick = p["session"], p["echo_tick"]
        self.tick_received = self.clock()

    def command(self, kind: str, payload: dict, *, seq: int | None = None) -> tuple[int, bytes]:
        if kind not in {"ping", "manual", "mission"} or not _payload(kind, payload):
            raise ProtocolError("invalid command payload")
        if not self.session or not 0 <= self.clock() - self.tick_received <= FRESH_MS / 1000:
            raise ProtocolError("receiver tick expired; establish a new session")
        if seq is None:
            seq = self.sequences[kind] + 1
        if not _integer(seq, 1):
            raise ProtocolError("invalid sequence")
        digest = payload_digest(payload)
        previous = self.outstanding.get(kind)
        if previous is None or previous[0] != seq:
            self.outstanding[kind] = (seq, self.clock(), digest)
        elif previous[2] != digest:
            raise ProtocolError("retry payload changed")
        self.sequences[kind] = max(self.sequences[kind], seq)
        return seq, encode(self.key, kind, self.session, seq, self.tick, payload)

    def response(self, wire: bytes, kind: str, seq: int, payload: dict) -> dict:
        p = decode(self.key, wire)
        expected = {"ping": "pong", "manual": "manual_ack", "mission": "ack"}[kind]
        result = p["payload"]
        pending = self.outstanding.get(kind)
        if (pending is None or pending[0] != seq or pending[2] != payload_digest(payload)
                or not 0 <= self.clock() - pending[1] <= FRESH_MS / 1000):
            raise ProtocolError("unmatched or expired response")
        if (p["type"] != expected or p["session"] != self.session or p["seq"] != seq
                or p["echo_tick"] < self.tick
                or result.get("applied") is not False
                or result.get("bench_only") is not True):
            raise ProtocolError("unexpected response")
        if kind == "ping":
            state = result.get("state")
            if (set(result) != {"state", *BENCH} or not isinstance(state, dict)
                    or set(state) != {"manual_held", "mission_queued"}
                    or type(state["manual_held"]) is not bool
                    or not _integer(state["mission_queued"], 0, MISSION_CAPACITY)):
                raise ProtocolError("invalid receiver state")
        elif (set(result) != {"status", "digest", *BENCH}
              or result["digest"] != payload_digest(payload)
              or not isinstance(result["status"], str)
              or result["status"] not in ({"QUEUED_BENCH", "BUSY"} if kind == "mission"
                                         else {"QUEUED_BENCH"})):
            raise ProtocolError("unexpected acknowledgement")
        # Anchor conservatively to the matched request's send time, rather
        # than accumulating network delay by anchoring to response receipt.
        self.tick, self.tick_received = p["echo_tick"], pending[1]
        return result


class BenchClient:
    """Single controller UDP client. Use a persistent instance for a control loop."""

    def __init__(self, key: bytes, host: str, port: int = PORT):
        family, self.address = literal_endpoint(host, port)
        self.protocol = ClientProtocol(key)
        self._mission_unknown = False
        self.socket = socket.socket(family, socket.SOCK_DGRAM)
        try:
            self.socket.connect(self.address)  # Kernel filters peer address, port and scope.
        except OSError:
            self.socket.close()
            raise

    def close(self) -> None:
        self.socket.close()

    def __enter__(self) -> BenchClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _receive(self, deadline: float) -> bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("bench link response timeout")
        self.socket.settimeout(remaining)
        return self.socket.recv(MAX_DATAGRAM + 1)

    def connect(self, timeout: float = 3.0) -> dict:
        # A prior standalone CLI process may still own the two-second lease.
        self.protocol = ClientProtocol(self.protocol.key)
        self._mission_unknown = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.socket.send(self.protocol.hello())
            attempt = min(deadline, time.monotonic() + 0.2)
            while time.monotonic() < attempt:
                try:
                    self.protocol.welcome(self._receive(attempt))
                    return self.ping(timeout=min(0.15, max(0.001, deadline - time.monotonic())))
                except (ProtocolError, TimeoutError, OSError):
                    continue
        raise TimeoutError("bench session handshake timed out")

    def _exchange(self, kind: str, payload: dict, timeout: float, retry: bool) -> dict:
        if not 0 < timeout <= 3:
            raise ValueError("response timeout must be in (0, 3] seconds")
        seq, wire = self.protocol.command(kind, payload)
        deadline = time.monotonic() + timeout
        while True:
            sent = time.monotonic()
            if sent >= deadline:
                raise TimeoutError("bench acceptance unknown; do not replay after reconnect")
            self.socket.send(wire)
            attempt = min(deadline, sent + 0.15) if retry else deadline
            while time.monotonic() < attempt:
                try:
                    result = self.protocol.response(self._receive(attempt), kind, seq, payload)
                    return result
                except (ProtocolError, TimeoutError):
                    continue
            if not retry or time.monotonic() >= deadline:
                raise TimeoutError("bench acceptance unknown; do not replay after reconnect")
            # Retries keep the mission identity. Refresh signed receiver ticks by
            # pinging while the previous tick is still fresh, never reconnecting.
            try:
                self.ping(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
                _, wire = self.protocol.command(kind, payload, seq=seq)
            except (ProtocolError, TimeoutError):
                # Without a fresh tick this same packet can only recover a
                # cached acceptance; it cannot introduce an aged new mission.
                pass
            time.sleep(max(0, min(sent + 0.2, deadline) - time.monotonic()))

    def ping(self, timeout: float = 0.2) -> dict:
        return self._exchange("ping", {}, timeout, False)

    def mission(self, op: int, args: bytes = b"", timeout: float = 1.0) -> dict:
        if self._mission_unknown:
            raise ProtocolError("previous mission acceptance unknown; explicit reconnect required")
        try:
            return self._exchange("mission", {"op": op, "args": args.hex()}, timeout, True)
        except (OSError, TimeoutError, ProtocolError):
            self._mission_unknown = True
            raise

    def manual(self, vector: tuple[float, float, float, float], hold: bool = True,
               timeout: float = 0.04) -> dict:
        return self._exchange("manual", dict(zip(("forward", "right", "up", "yaw"), vector),
                                              hold=hold), timeout, False)


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return value


def _vector(text: str) -> tuple[float, float, float, float]:
    try:
        vector = tuple(float(part) for part in text.split(","))
        if len(vector) != 4 or not all(math.isfinite(x) and -1 <= x <= 1 for x in vector):
            raise ValueError
        return vector
    except ValueError as exc:
        raise argparse.ArgumentTypeError("vector must be four numbers in [-1,1]: f,r,u,y") from exc


def _receive_cli(args: argparse.Namespace, key: bytes) -> None:
    family, endpoint = literal_endpoint(args.bind, args.port)
    bind = args.bind
    receiver = Receiver(key)
    stop = threading.Event()
    errors: list[Exception] = []
    fd = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as log, socket.socket(
            family, socket.SOCK_DGRAM) as sock:
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind(endpoint)
        sock.settimeout(0.05)

        def write_events() -> None:
            try:
                while not stop.is_set():
                    for event in receiver.drain_events():
                        log.write(canonical(event).decode("ascii") + "\n")
                    log.flush()
                    stop.wait(0.02)
                for event in receiver.drain_events():
                    log.write(canonical(event).decode("ascii") + "\n")
                log.flush()
            except Exception as exc:
                errors.append(exc)
                stop.set()

        writer = threading.Thread(target=write_events, name="bench-event-writer", daemon=True)
        writer.start()
        print(json.dumps({"status": "LISTENING_BENCH", "bind": bind, "port": args.port, **BENCH}),
              flush=True)
        end = time.monotonic() + args.seconds if args.seconds else math.inf
        try:
            while time.monotonic() < end and not stop.is_set():
                receiver.poll()
                try:
                    wire, source = sock.recvfrom(MAX_DATAGRAM + 1)
                except socket.timeout:
                    continue
                response = receiver.handle(wire, source)
                if response is not None:
                    sock.sendto(response, source)
        finally:
            receiver.close()
            stop.set()
            writer.join(timeout=1)
        if writer.is_alive() or errors:
            raise OSError("bench event writer failed or did not stop")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="create a new private 32-byte key")
    keygen.add_argument("--key-file", required=True)
    receive = commands.add_parser("receive", help="bench receiver; JSONL events, no actuators")
    receive.add_argument("--key-file", required=True)
    receive.add_argument("--bind", required=True, help="explicit IPv4/IPv6 literal (link-local needs %interface)")
    receive.add_argument("--port", type=int, default=PORT)
    receive.add_argument("--log", required=True, help="new JSONL file; never overwritten")
    receive.add_argument("--seconds", type=_positive_seconds)
    for name in ("ping", "mission", "manual-demo"):
        command = commands.add_parser(name)
        command.add_argument("--key-file", required=True)
        command.add_argument("--host", required=True, help="receiver IPv4/IPv6 literal (link-local needs %interface)")
        command.add_argument("--port", type=int, default=PORT)
        if name == "mission":
            command.add_argument("--op", type=int, required=True, help="opaque uint8; mapping pending")
            command.add_argument("--args", default="", help="hex bytes, maximum 48 bytes")
        if name == "manual-demo":
            command.add_argument("--vector", type=_vector, required=True, help="forward,right,up,yaw")
            command.add_argument("--seconds", type=_positive_seconds, default=1.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "keygen":
            generate_key(args.key_file)
            print("Created private bench key file.")
            return 0
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        key = load_key(args.key_file)
        if args.command == "receive":
            _receive_cli(args, key)
            return 0
        if args.command == "manual-demo" and args.seconds > 10:
            raise ValueError("manual demo maximum is 10 seconds")
        if args.command == "mission" and not _payload("mission", {"op": args.op, "args": args.args}):
            raise ValueError("mission requires uint8 op and lowercase hex args of at most 48 bytes")
        with BenchClient(key, args.host, args.port) as client:
            result = client.connect()
            if args.command == "mission":
                result = client.mission(args.op, bytes.fromhex(args.args))
            elif args.command == "manual-demo":
                end = time.monotonic() + args.seconds
                try:
                    while time.monotonic() < end:
                        start = time.monotonic()
                        try:
                            result = client.manual(args.vector)
                        except TimeoutError:
                            pass  # Discard this setpoint; next iteration gets a new sequence.
                        time.sleep(max(0, min(0.05 - (time.monotonic() - start),
                                              end - time.monotonic())))
                finally:
                    result = client.manual((0, 0, 0, 0), hold=False)
            print(json.dumps(result, sort_keys=True))
            return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, TimeoutError) as exc:
        # Do not print arbitrary exception data which could contain a key or packet.
        print(f"Bench link failed ({type(exc).__name__}); no actuation is implemented.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
