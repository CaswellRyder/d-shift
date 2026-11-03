"""Pi-side setup helper, streamed over SSH by flight_lab.py. Standard library only."""

import base64
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

PROC_ROOT = Path('/proc')


class SetupError(Exception):
    pass


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ('..', '') for p in path.parts):
        raise SetupError('Unsafe path in bundle manifest')
    return path


def checked_file(root, relative):
    path = Path(root).joinpath(*safe_relative(relative).parts)
    try:
        path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        raise SetupError('Bundle file resolves outside its installation')
    if path.is_symlink() or not path.is_file():
        raise SetupError('Bundle file is missing: ' + relative)
    return path


def verify_files(root, manifest, code_only=False):
    for relative, expected in manifest['files'].items():
        if code_only and not relative.startswith(('src/', 'scripts/', 'configs/')):
            continue
        path = checked_file(root, relative)
        if digest(path) != expected:
            raise SetupError('Installed file differs from this kit: ' + relative)


def argument(argv, name):
    if argv.count(name) != 1:
        raise SetupError('Running viewer has an unexpected ' + name + ' argument')
    index = argv.index(name) + 1
    if index >= len(argv):
        raise SetupError('Running viewer argument has no value: ' + name)
    return Path(argv[index])


DEFAULT_BALLOON_LIMIT = 4  # pi_flight_lab.py default when --balloon-limit is absent.


def validate_process(root, manifest, expected_port=None, expected_balloon_limit=None):
    root = Path(root)
    try:
        pid = int((root / 'server.pid').read_text())
        process = PROC_ROOT / str(pid)
        if (process / 'stat').read_text().split()[2] == 'Z':
            return None
        argv = [x.decode('utf-8') for x in (process / 'cmdline').read_bytes().split(b'\0') if x]
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    script = str(root / 'scripts/pi_flight_lab.py')
    if script not in argv:
        return None
    try:
        actual_port = int(str(argument(argv, '--port'))) if '--port' in argv else 8770
    except (TypeError, ValueError):
        raise SetupError('Running viewer has an invalid port argument')
    if expected_port is not None and actual_port != expected_port:
        raise SetupError('Known Pi viewer is running on port %d, but this connection requests %d; '
                         'match connection.json to the existing port before retrying' %
                         (actual_port, expected_port))
    try:
        actual_limit = (int(str(argument(argv, '--balloon-limit'))) if '--balloon-limit' in argv
                        else DEFAULT_BALLOON_LIMIT)
    except (TypeError, ValueError):
        raise SetupError('Running viewer has an invalid balloon limit argument')
    if expected_balloon_limit is not None and actual_limit != expected_balloon_limit:
        raise SetupError('Known Pi viewer is running with balloon_limit %d, but this connection requests %d; '
                         'stop that viewer (after any capture is saved) or match connection.json before retrying' %
                         (actual_limit, expected_balloon_limit))
    verify_files(root, manifest, code_only=True)
    for kind in ('balloon', 'goal'):
        model = argument(argv, '--' + kind + '-model')
        relative = manifest['model_files'][kind]
        for actual, expected_relative in ((model, relative),
                                         (model.with_suffix('.json'), str(PurePosixPath(relative).with_suffix('.json')))):
            if expected_relative not in manifest['files'] or digest(actual) != manifest['files'][expected_relative]:
                raise SetupError('Running viewer uses a different ' + kind + ' model or metadata')
    # Read only the named runtime setting. Do not print or retain process environment.
    runtime_path = None
    for entry in (process / 'environ').read_bytes().split(b'\0'):
        if entry.startswith(b'DTR_TFLITE_LIBRARY='):
            runtime_path = Path(os.fsdecode(entry.split(b'=', 1)[1]))
            break
    if runtime_path is None or digest(runtime_path) != manifest['files'][manifest['runtime_file']]:
        raise SetupError('Running viewer uses a different native runtime')
    for flag, relative in (('--camera-mount', 'configs/camera_mount.json'),
                           ('--target-sizes', 'configs/target_sizes.json')):
        if flag in argv and digest(argument(argv, flag)) != manifest['files'][relative]:
            raise SetupError('Running viewer uses different coordinate configuration')
    if '--lens-calibration' in argv:
        raise SetupError('Running viewer has an external lens calibration not contained in this kit')
    return pid


def read_viewer_state(port):
    try:
        with urlopen('http://127.0.0.1:%d/api/state' % port, timeout=3) as response:
            state = json.load(response)
    except URLError as error:
        reason = error.reason
        if isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
            return None
        raise SetupError('The Pi port is occupied or not responding as the lab viewer')
    except (OSError, ValueError):
        raise SetupError('The Pi port is occupied or not responding as the lab viewer')
    if (not isinstance(state, dict) or state.get('source') != 'camera'
            or not isinstance(state.get('capture'), dict)):
        raise SetupError('The Pi port belongs to another application; it was not changed')
    return state


def probe(payload):
    config, manifest = payload['config'], payload['manifest']
    roots = [config['remote_root']]
    if config.get('legacy_root'):
        roots.append(config['legacy_root'])
    state = read_viewer_state(config['port'])
    if state is not None:
        if state['capture'].get('state') in ('capturing', 'exporting'):
            raise SetupError('Wait until the current take is saved before reconnecting or downloading')
        root = state.get('bundle')
        if root not in roots:
            raise SetupError('Another Pi bundle owns this port; it was not stopped or changed')
        if validate_process(root, manifest, config['port'],
                            config.get('balloon_limit', DEFAULT_BALLOON_LIMIT)) is None:
            raise SetupError('The viewer PID identity could not be verified; it was not changed')
        return dict(kind='existing', root=root, ready=state.get('ready', False),
                    error=state.get('error'), capture_state=state['capture'].get('state'))
    for root in roots:
        if validate_process(root, manifest, config['port'],
                            config.get('balloon_limit', DEFAULT_BALLOON_LIMIT)) is not None:
            return dict(kind='starting', root=root, ready=False, error=None)
    return dict(kind='absent', root=config['remote_root'])


def install(payload):
    config, manifest = payload['config'], payload['manifest']
    root, archive_path = Path(config['remote_root']), Path(payload['archive_path'])
    if digest(archive_path) != manifest['archive_sha256']:
        raise SetupError('Transferred bundle archive failed its SHA-256 check')
    if root.exists():
        verify_files(root, manifest)
        archive_path.unlink()
        return dict(kind='installed', root=str(root), reused=True)
    if not root.parent.is_dir():
        raise SetupError('The installation parent directory does not exist: ' + str(root.parent))
    staging = Path(tempfile.mkdtemp(prefix='.' + root.name + '-install-', dir=str(root.parent)))
    try:
        with tarfile.open(archive_path, 'r:gz') as archive:
            members = archive.getmembers()
            actual_files = set()
            for member in members:
                safe_relative(member.name)
                if not (member.isfile() or member.isdir()):
                    raise SetupError('Bundle archive contains a link or special file')
                if member.isfile():
                    if member.name in actual_files:
                        raise SetupError('Bundle archive repeats a file')
                    actual_files.add(member.name)
            if actual_files != set(manifest['files']):
                raise SetupError('Bundle archive files do not match its manifest')
            # All paths and member types were checked; staging is a new private directory.
            archive.extractall(staging)
        verify_files(staging, manifest)
        for path in staging.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)
        (staging / 'handoff-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if root.exists():
            raise SetupError('Installation appeared concurrently; existing files were not changed')
        staging.rename(root)
        return dict(kind='installed', root=str(root), reused=False)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        # The uploaded temporary archive belongs to this setup operation.
        archive_path.unlink(missing_ok=True)


def start(payload):
    existing = probe(payload)
    if existing['kind'] != 'absent':
        return existing
    root = Path(payload['config']['remote_root'])
    verify_files(root, payload['manifest'])
    with (root / '.startup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SetupError('Pi startup is already in progress; wait, then run Start Flight Lab again')
        # Recheck after acquiring the cross-process lock. A concurrent invocation
        # may already have published its verified PID while the camera warms up.
        return _start_locked(payload)


def _start_locked(payload):
    config, manifest = payload['config'], payload['manifest']
    existing = probe(payload)
    if existing['kind'] != 'absent':
        return existing
    root = Path(config['remote_root'])
    verify_files(root, manifest)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(root / 'src') + ':' + str(root),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               DTR_TFLITE_LIBRARY=str(root / manifest['runtime_file']))
    command = ['python3', '-u', str(root / 'scripts/pi_flight_lab.py'),
               '--balloon-model', str(root / manifest['model_files']['balloon']),
               '--goal-model', str(root / manifest['model_files']['goal']),
               '--balloon-limit', str(config.get('balloon_limit', DEFAULT_BALLOON_LIMIT)),
               '--output', str(root / 'sessions'), '--port', str(config['port']),
               '--wall-clock-offset-s', str(payload['wall_clock_offset_s'])]
    with (root / 'server.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'server.pid').write_text(str(process.pid))
    return dict(kind='starting', root=str(root), pid=process.pid)


def main():
    try:
        payload = json.loads(base64.b64decode(sys.argv[1]))
        action = payload['action']
        if action == 'epoch':
            import time
            result = dict(epoch=time.time())
        elif action == 'probe':
            result = probe(payload)
        elif action == 'install':
            result = install(payload)
        elif action == 'start':
            result = start(payload)
        else:
            raise SetupError('Unsupported setup action')
        print(json.dumps(dict(ok=True, result=result)))
    except (SetupError, OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(json.dumps(dict(ok=False, error=str(error))))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
