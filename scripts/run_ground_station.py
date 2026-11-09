"""Keep the local bench console and optional Wi-Fi capture tunnel running."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time


class StartupError(Exception):
    """A local startup requirement was not met."""


@dataclass(frozen=True)
class Config:
    binary: Path
    key_file: Path
    runtime_dir: Path
    capture_dir: Path | None = None
    capture_python: Path | None = None
    capture_processor: Path | None = None
    ssh_identity: Path | None = None
    wifi_interface: str | None = None


def absolute_path(value, name, *, executable=False, private=False, directory=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise StartupError(f'{name} must be an absolute path')
    path = Path(value)
    if directory:
        if path.exists() and not path.is_dir():
            raise StartupError(f'{name} must be a directory')
        return path
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise StartupError(f'{name} must name an existing ' +
                           ('executable' if executable else 'file'))
    if private and (path.stat().st_mode & 0o077):
        raise StartupError(f'{name} must not be accessible to other users')
    return path


def load_config(filename):
    path = absolute_path(filename, 'config')
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise StartupError('Cannot read the station configuration') from error
    fields = set(Config.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) - fields:
        raise StartupError('Configuration contains unknown fields')
    for required in ('binary', 'key_file', 'runtime_dir'):
        if required not in raw:
            raise StartupError(f'Configuration requires {required}')
    parsed = {}
    for name, value in raw.items():
        if name == 'wifi_interface':
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,32}', value):
                raise StartupError('wifi_interface is not a valid interface name')
            parsed[name] = value
        else:
            parsed[name] = absolute_path(
                value, name, executable=name in ('binary', 'capture_python'),
                private=name in ('key_file', 'ssh_identity'),
                directory=name in ('runtime_dir', 'capture_dir'))
    captures = ('capture_dir', 'capture_python', 'capture_processor')
    if any(name in parsed for name in captures) and not all(name in parsed for name in captures):
        raise StartupError('Capture directory, Python and processor must be configured together')
    if parsed['key_file'].stat().st_size != 32:
        raise StartupError('key_file must contain exactly 32 raw bytes')
    if parsed.get('ssh_identity') and parsed['ssh_identity'].stat().st_size == 0:
        raise StartupError('ssh_identity is empty')
    return Config(**parsed)


def prepare_runtime(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise StartupError('runtime_dir must be a private directory owned by this user')


@contextmanager
def single_instance(runtime):
    prepare_runtime(runtime)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(runtime / 'station.lock', flags, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StartupError('A ground station already owns this runtime directory') from error
        yield
    finally:
        os.close(fd)


def station_command(config, session):
    command = [str(config.binary), '--peer', '192.168.50.1:47110',
               '--key-file', str(config.key_file), '--listen', '127.0.0.1:8782',
               '--journal', str(session / 'laptop.jsonl'), '--seconds', '0']
    if config.wifi_interface:
        command += ['--interface', config.wifi_interface]
    if config.capture_dir:
        command += ['--lab-url', 'http://127.0.0.1:8771',
                    '--capture-dir', str(config.capture_dir),
                    '--capture-python', str(config.capture_python),
                    '--capture-processor', str(config.capture_processor)]
    return command


def tunnel_command(config):
    if config.ssh_identity is None:
        return None
    command = ['/usr/bin/ssh', '-N', '-T', '-n', '-o', 'BatchMode=yes',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=vision.local',
               '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=5',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
               '-o', 'ExitOnForwardFailure=yes', '-o', 'ControlMaster=no',
               '-o', 'ControlPath=none', '-i', str(config.ssh_identity)]
    if config.wifi_interface:
        command += ['-B', config.wifi_interface]
    return command + ['-L', '127.0.0.1:8771:127.0.0.1:8771', 'pacman@192.168.50.1']


def stop_child(process):
    """Stop only a process group created by this launcher, including descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def run(config, stop, *, popen=subprocess.Popen, cleanup=stop_child, clock=time.monotonic):
    """Supervise local children until signalled. Never issue a flight command."""
    with single_instance(config.runtime_dir):
        session = Path(tempfile.mkdtemp(prefix='session-', dir=config.runtime_dir))
        laptop = tunnel = None
        next_tunnel = 0.0
        tunnel_started = 0.0
        backoff = 1.0
        next_laptop = 0.0
        laptop_started = 0.0
        laptop_backoff = 1.0
        ssh = tunnel_command(config)
        with (session / 'laptop.log').open('xb') as laptop_log, \
                (session / 'tunnel.log').open('xb') as tunnel_log:
            # Session directory is 0700; file modes remain private even with a broad caller umask.
            os.chmod(laptop_log.name, 0o600)
            os.chmod(tunnel_log.name, 0o600)
            try:
                while not stop.is_set():
                    now = clock()
                    if laptop is not None and laptop.poll() is not None:
                        cleanup(laptop)
                        laptop = None
                        if now - laptop_started >= 30:
                            laptop_backoff = 1.0
                        next_laptop = now + laptop_backoff
                        laptop_backoff = min(laptop_backoff * 2, 30.0)
                    if laptop is None and now >= next_laptop:
                        attempt = Path(tempfile.mkdtemp(prefix='link-', dir=session))
                        laptop = popen(station_command(config, attempt),
                                       stdin=subprocess.DEVNULL, stdout=laptop_log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       close_fds=True)
                        laptop_started = now
                    if tunnel is not None and tunnel.poll() is not None:
                        cleanup(tunnel)
                        tunnel = None
                        if now - tunnel_started >= 30:
                            backoff = 1.0
                        next_tunnel = now + backoff
                        backoff = min(backoff * 2, 30.0)
                    if ssh is not None and tunnel is None and now >= next_tunnel:
                        tunnel = popen(ssh, stdin=subprocess.DEVNULL, stdout=tunnel_log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       close_fds=True)
                        tunnel_started = now
                    stop.wait(0.25)
                return 0
            finally:
                if tunnel is not None:
                    cleanup(tunnel)
                if laptop is not None:
                    cleanup(laptop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Absolute private station JSON path')
    args = parser.parse_args()
    os.umask(0o077)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        return run(load_config(args.config), stop)
    except (StartupError, OSError) as error:
        # Never include subprocess arguments, environment variables or key contents.
        print(f'Ground station could not start: {error}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
