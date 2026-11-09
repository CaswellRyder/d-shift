"""Local-only setup and supervision for the transferred MacBook kit."""

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading


KIT = Path(__file__).resolve().parents[1]
ARCHES = {'arm64': 'arm64', 'aarch64': 'arm64', 'x86_64': 'amd64', 'amd64': 'amd64'}
PRIVATE_NAMES = ('ground.key', 'capture_ed25519', 'known_hosts', 'wifi-password.txt')


def architecture():
    try:
        return ARCHES[platform.machine()]
    except KeyError as error:
        raise ValueError('Unsupported Mac processor') from error


def paths(kit=KIT, arch=None):
    arch = architecture() if arch is None else arch
    return dict(site=kit / '.runtime' / ('site-' + arch),
                python=kit / '.runtime' / ('python-' + arch) / 'python/bin/python3.11',
                source=kit / 'source/src')


def checked_directory(path, *, create=False):
    if path.is_symlink():
        raise ValueError('A kit runtime/private directory is a symbolic link')
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ValueError('Kit directory must be owned by the current user')
    path.chmod(0o700)


def import_private(kit=KIT, source=None):
    source = kit.with_name(kit.name + '-PRIVATE') if source is None else Path(source)
    target = kit / 'private'
    if not target.exists():
        checked_directory(target, create=True)
    else:
        checked_directory(target)
    missing = []
    for name in PRIVATE_NAMES:
        destination = target / name
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                raise ValueError('Private credential must be a regular file')
            destination.chmod(0o600)
            continue
        origin = source / 'private' / name
        if origin.is_symlink() or not origin.is_file():
            missing.append(name)
            continue
        # Exclusive creation never replaces credentials from an earlier pairing.
        with destination.open('xb') as stream:
            os.chmod(destination, 0o600)
            stream.write(origin.read_bytes())
    if missing:
        raise ValueError('Extract the separate PRIVATE ZIP next to the program kit before setup')
    if (target / 'ground.key').stat().st_size != 32:
        raise ValueError('Pairing key must contain exactly 32 raw bytes')
    if not (target / 'capture_ed25519').stat().st_size:
        raise ValueError('Capture SSH identity is empty')
    return target


def python_environment(kit=KIT, arch=None):
    selected = paths(kit, arch)
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONINSPECT'):
        env.pop(key, None)
    env['PYTHONPATH'] = os.pathsep.join((str(selected['site']), str(selected['source'])))
    env['PYTHONNOUSERSITE'] = '1'
    return env


def setup(kit=KIT, arch=None, *, runner=subprocess.run):
    selected = paths(kit, arch)
    arch = architecture() if arch is None else arch
    import_private(kit)
    if not selected['python'].is_file():
        raise ValueError('Bundled Python has not been extracted; use Setup MacBook.command')
    site = selected['site']
    if site.is_symlink():
        raise ValueError('Python package directory must not be a symbolic link')
    if not site.exists():
        temporary = Path(tempfile.mkdtemp(prefix='packages-', dir=kit / '.runtime'))
        try:
            command = [str(selected['python']), '-I', '-m', 'pip', '--isolated',
                       '--disable-pip-version-check', 'install', '--no-index', '--no-deps',
                       '--no-cache-dir', '--only-binary=:all:', '--require-hashes',
                       '--find-links', str(kit / 'offline' / ('wheels-' + arch)),
                       '--target', str(temporary), '-r',
                       str(kit / 'offline' / ('requirements-' + arch + '.txt'))]
            runner(command, check=True, env=python_environment(kit, arch))
            temporary.rename(site)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    check = ('import sys; sys.path[:0]=sys.argv[1:]; '
             'import numpy, cv2, PIL, dtr.raw_capture; '
             'assert numpy.__version__ == "1.26.4"; '
             'assert cv2.__version__ == "4.11.0"; '
             'assert PIL.__version__ == "11.1.0"; '
             'print("Offline capture processor imports passed")')
    runner([str(selected['python']), '-I', '-c', check,
            str(site), str(selected['source'])], check=True,
           env=python_environment(kit, arch))
    print('Setup complete. Join DTR-Drone-Bench, then run Start Ground Station.command.')
    print('No camera, Wi-Fi profile, login job or motor command was started by setup.')


def wifi_interface(*, runner=subprocess.run):
    result = runner(['/usr/sbin/networksetup', '-listallhardwareports'],
                    capture_output=True, text=True, check=True)
    wanted = False
    for line in result.stdout.splitlines():
        if line.startswith('Hardware Port:'):
            wanted = line.partition(':')[2].strip() in ('Wi-Fi', 'AirPort')
        elif wanted and line.startswith('Device:'):
            interface = line.partition(':')[2].strip()
            if re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,31}', interface):
                return interface
    raise ValueError('Cannot identify this MacBook Wi-Fi interface')


def private_tunnel(original_tunnel, config, known_hosts):
    command = original_tunnel(config)
    if command is not None:
        quoted_path = '"' + str(known_hosts).replace('\\', '\\\\').replace('"', '\\"') + '"'
        command[1:1] = ['-F', '/dev/null', '-o', 'UserKnownHostsFile=' + quoted_path,
                        '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'UpdateHostKeys=no',
                        '-o', 'IdentityAgent=none']
    return command


def load_station(kit=KIT):
    filename = kit / 'source/scripts/run_ground_station.py'
    spec = importlib.util.spec_from_file_location('transferred_ground_station', filename)
    station = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = station
    spec.loader.exec_module(station)
    return station


def start(kit=KIT):
    arch = architecture()
    selected = paths(kit, arch)
    if not selected['site'].is_dir():
        raise ValueError('Run Setup MacBook.command first')
    private = import_private(kit)
    checked_directory(kit / '.runtime', create=True)
    # Binding preflight detects an existing station without taking it over.
    for port in (8782, 8771):
        with socket.socket() as probe:
            # Ignore closed TCP sessions while still refusing a live listener.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', port))
            except OSError as error:
                raise ValueError(f'Port {port} is already in use; stop its owning station first') from error
    station = load_station(kit)
    runtime = kit / '.runtime/station'
    captures = kit / 'Captures'
    checked_directory(runtime, create=True)
    checked_directory(captures, create=True)
    config = station.Config(binary=kit / 'bin' / ('laptop-link-' + arch),
                            key_file=private / 'ground.key', runtime_dir=runtime,
                            capture_dir=captures,
                            capture_python=kit / 'tools/capture-python',
                            capture_processor=kit / 'source/scripts/process_raw_capture.py',
                            ssh_identity=private / 'capture_ed25519',
                            wifi_interface=wifi_interface())
    original_tunnel = station.tunnel_command

    def pinned_tunnel(value):
        # Call the original function rather than recursively calling this adapter.
        return private_tunnel(original_tunnel, value, private / 'known_hosts')

    station.tunnel_command = pinned_tunnel
    os.environ.update(python_environment(kit, arch))
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: stop.set())
    print('Open http://localhost:8782. The console stays available while waiting for Wi-Fi.')
    print('Join DTR-Drone-Bench for the Pi link; reconnect requires fresh Resume.')
    print('Leave this Terminal open. Control-C stops only this station and its tunnel.')
    print('Manual bench controls. Launch is disabled; inspect Controller telemetry for ESP32 status.')
    try:
        return station.run(config, stop)
    except station.StartupError as error:
        raise ValueError(str(error)) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('setup', 'start'))
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if sys.version_info[:2] != (3, 11):
            raise ValueError('The bundled Python 3.11 runtime is required')
        if args.operation == 'setup':
            setup()
            return 0
        return start()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # Avoid command dumps, environments and credential contents.
        print('MacBook station could not start: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
