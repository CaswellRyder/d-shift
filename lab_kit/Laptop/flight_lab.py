"""Portable laptop entry point. Python 3.9+, SSH/SCP and a browser; no pip packages."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = re.compile(r'\d{8}T\d{6}-[0-9a-f]{8}')


class LabError(Exception):
    pass


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def relative_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise LabError('Unsafe package manifest path')
    target = Path(root).joinpath(*path.parts)
    try:
        target.resolve().relative_to(Path(root).resolve())
    except ValueError:
        raise LabError('Package path resolves outside this folder')
    return target


def check_package(root=ROOT):
    sums_path = Path(root) / 'SHA256SUMS.json'
    if not sums_path.is_file():
        raise LabError('SHA256SUMS.json is missing. Copy the entire lab kit folder.')
    sums = json.loads(sums_path.read_text())
    expected = sums.get('files', sums)
    if not isinstance(expected, dict) or not expected:
        raise LabError('The package checksum file is invalid')
    count = 0
    for relative, checksum in expected.items():
        if not isinstance(relative, str) or not isinstance(checksum, str):
            raise LabError('The package checksum file is invalid')
        path = relative_path(root, relative)
        if not path.is_file() or path.is_symlink() or digest(path) != checksum:
            raise LabError('Package integrity check failed: ' + relative)
        count += 1
    bundle = Path(root) / 'Pi Bundle'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('schema') != 1 or not isinstance(manifest.get('files'), dict):
        raise LabError('The Pi bundle manifest is invalid')
    if digest(bundle / 'pi-bundle.tar.gz') != manifest.get('archive_sha256'):
        raise LabError('The Pi bundle archive failed its SHA-256 check')
    for relative, checksum in manifest['files'].items():
        relative_path(root, relative)
        if not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum):
            raise LabError('The Pi bundle checksum table is invalid')
    for kind in ('balloon', 'goal'):
        if manifest['model_files'][kind] not in manifest['files']:
            raise LabError('The Pi bundle is missing its ' + kind + ' model')
    if manifest['runtime_file'] not in manifest['files']:
        raise LabError('The Pi bundle is missing its native runtime')
    lens = manifest.get('lens_calibration_file')
    if lens is not None and (lens != 'configs/lens-calibration.json' or lens not in manifest['files']):
        raise LabError('The Pi bundle has an invalid pinned lens calibration')
    return manifest, count


def load_config(root=ROOT):
    config = json.loads((Path(root) / 'connection.json').read_text())
    if not isinstance(config, dict):
        raise LabError('connection.json must contain an object')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', str(config.get('host', ''))):
        raise LabError('Invalid Pi hostname in connection.json')
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', str(config.get('user', ''))):
        raise LabError('Invalid Pi user in connection.json')
    port = config.get('port')
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise LabError('The lab port must be an integer from 1024 to 65535')
    limit = config.get('balloon_limit', 4)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
        raise LabError('The balloon limit must be an integer from 1 to 64')
    for key in ('remote_root', 'legacy_root'):
        value = config.get(key)
        if value is None and key == 'legacy_root':
            continue
        if (not isinstance(value, str) or not re.fullmatch(r'/[A-Za-z0-9_./-]+', value)
                or '..' in PurePosixPath(value).parts or len(PurePosixPath(value).parts) < 4):
            raise LabError('Invalid Pi installation path: ' + key)
    return config


def read_state(url):
    try:
        with urlopen(url + '/api/state', timeout=4) as response:
            result = json.load(response)
        return result if isinstance(result, dict) else None
    except (OSError, ValueError, URLError):
        return None


def remote_command(payload):
    encoded = base64.b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode('ascii')
    return 'python3 - ' + shlex.quote(encoded)


class Connection:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.destination = config['user'] + '@' + config['host']
        self.url = 'http://127.0.0.1:%d' % config['port']
        self.state_path = self.root / 'Laptop' / '.runtime.json'
        self.control = None
        self.selected_root = None

    def _save(self):
        state = dict(control=self.control, destination=self.destination,
                     port=self.config['port'], selected_root=self.selected_root)
        temporary = self.state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.chmod(0o600)
        temporary.replace(self.state_path)

    def connect(self):
        if shutil.which('ssh') is None or shutil.which('scp') is None:
            raise LabError('This laptop needs the macOS SSH and SCP command-line tools')
        try:
            state = json.loads(self.state_path.read_text())
            control = Path(state['control'])
            private_parent = control.parent
            owned = (private_parent.name.startswith('dtr-flight-lab-ssh-')
                     and private_parent.parent == Path(tempfile.gettempdir())
                     and private_parent.stat().st_uid == os.getuid()
                     and not private_parent.is_symlink()
                     and private_parent.stat().st_mode & 0o077 == 0)
            if (owned and state.get('destination') == self.destination
                    and state.get('port') == self.config['port']):
                check = subprocess.run(['ssh', '-S', str(control), '-O', 'check', self.destination],
                                       capture_output=True, text=True)
                if check.returncode == 0:
                    self.control = str(control)
                    self.selected_root = state.get('selected_root')
                    return
        except (OSError, ValueError, KeyError, TypeError):
            pass
        with socket.socket() as test:
            test.settimeout(.5)
            if test.connect_ex(('127.0.0.1', self.config['port'])) == 0:
                raise LabError('Local port %d is already in use. Its application was left running.' % self.config['port'])
        private_parent = Path(tempfile.mkdtemp(prefix='dtr-flight-lab-ssh-'))
        private_parent.chmod(0o700)
        self.control = str(private_parent / 'control')
        fingerprint = self.config.get('expected_host_ed25519_fingerprint')
        if fingerprint:
            print('Expected Pi ED25519 host fingerprint: ' + str(fingerprint), flush=True)
            print('On first connection, compare the SSH prompt with that fingerprint before accepting.', flush=True)
        print('Connecting to the Pi. SSH may ask for its password in this terminal.', flush=True)
        subprocess.run(['ssh', '-M', '-S', self.control, '-o', 'ControlPersist=7200',
                        '-o', 'ExitOnForwardFailure=yes', '-o', 'StrictHostKeyChecking=ask',
                        '-o', 'ConnectTimeout=10', '-fN', '-L',
                        '127.0.0.1:%d:127.0.0.1:%d' % (self.config['port'], self.config['port']),
                        self.destination], check=True)
        self._save()

    def remote(self, action, manifest, **extra):
        payload = dict(action=action, config=self.config, manifest=manifest, **extra)
        code = (self.root / 'Laptop' / 'pi_helper.py').read_text()
        command = ['ssh', '-S', self.control, '-o', 'BatchMode=yes', self.destination,
                   remote_command(payload)]
        result = subprocess.run(command, input=code, text=True, capture_output=True)
        try:
            envelope = json.loads(result.stdout)
        except ValueError:
            raise LabError('Pi setup did not return a valid response. ' + result.stderr.strip())
        if result.returncode != 0 or not envelope.get('ok'):
            raise LabError(envelope.get('error') or 'Pi setup failed')
        return envelope['result']

    def remember_root(self, remote_root):
        self.selected_root = remote_root
        self._save()

    def prepare(self, manifest, install_if_missing):
        result = self.remote('probe', manifest)
        if result['kind'] == 'absent':
            if not install_if_missing:
                raise LabError('No running lab viewer was found. Run Start Flight Lab first.')
            archive_path = str(PurePosixPath(self.config['remote_root']).parent /
                               ('.flight-lab-upload-' + uuid.uuid4().hex + '.tar.gz'))
            print('Installing the verified Pi bundle in a new folder. Existing bundles stay in place.', flush=True)
            subprocess.run(['scp', '-o', 'ControlPath=' + self.control, '-o', 'BatchMode=yes',
                            str(self.root / 'Pi Bundle' / 'pi-bundle.tar.gz'),
                            self.destination + ':' + archive_path], check=True)
            self.remote('install', manifest, archive_path=archive_path)
            before = time.time()
            clock = self.remote('epoch', manifest)
            offset = (before + time.time()) / 2 - clock['epoch']
            result = self.remote('start', manifest, wall_clock_offset_s=offset)
        self.remember_root(result['root'])
        if result.get('error'):
            raise LabError('Pi viewer reports: ' + result['error'])
        print('Waiting for the Pi camera and models. ' + self.url, flush=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = read_state(self.url)
            if state is not None:
                if state.get('bundle') != self.selected_root or state.get('source') != 'camera':
                    raise LabError('The forwarded port changed applications; no process was stopped')
                if state.get('error'):
                    raise LabError('Pi viewer reports: ' + state['error'])
                if state.get('ready') or state.get('capture', {}).get('state') in ('capturing', 'exporting'):
                    return state
            time.sleep(.5)
        raise LabError('Viewer did not become ready. On the Pi, inspect ' +
                       self.selected_root + '/server.log. No existing process was stopped.')

    def download(self, state):
        if state.get('capture', {}).get('state') in ('capturing', 'exporting'):
            raise LabError('Wait until the current capture is saved before downloading')
        sessions = state.get('sessions', [])
        if not sessions:
            print('No completed lab takes are listed. Nothing downloaded.', flush=True)
            return None
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
        output = self.root / 'Captures' / name
        output.mkdir(parents=True)
        index = dict(created_at=datetime.now(timezone.utc).isoformat(), pi_root=self.selected_root,
                     scope='Completed takes currently listed by the viewer (at most the latest 100); all takes remain on the Pi',
                     complete=False, files=[])
        (output / 'download-index.json').write_text(json.dumps(index, indent=2) + '\n')
        try:
            for session in sessions:
                ident = session.get('id', '')
                if not SESSION_ID.fullmatch(ident):
                    raise LabError('The Pi returned an invalid capture identifier')
                path = output / (ident + '.zip')
                partial = path.with_suffix('.zip.partial')
                try:
                    # The server gates these transfers against timed captures. SCP would bypass that gate.
                    with urlopen(self.url + '/sessions/' + ident + '/download.zip', timeout=120) as source:
                        expected = source.headers.get('Content-Length')
                        copied = 0
                        with partial.open('xb') as destination:
                            while True:
                                block = source.read(1024 * 1024)
                                if not block:
                                    break
                                destination.write(block)
                                copied += len(block)
                    if expected is not None and copied != int(expected):
                        raise LabError('Capture download was interrupted: ' + ident)
                    partial.replace(path)
                except Exception:
                    # A partial download is retained and clearly named; it is never presented as complete.
                    raise
                index['files'].append(dict(id=ident, path=path.name, bytes=copied, sha256=digest(path)))
                (output / 'download-index.json').write_text(json.dumps(index, indent=2) + '\n')
                print('Saved ' + path.name, flush=True)
            index['complete'] = True
            (output / 'download-index.json').write_text(json.dumps(index, indent=2) + '\n')
            return output
        except Exception as error:
            index['error'] = str(error)
            (output / 'download-index.json').write_text(json.dumps(index, indent=2) + '\n')
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check-package', action='store_true', help='Verify local files only; no Pi connection')
    action.add_argument('--download', action='store_true', help='Download listed completed takes as ZIPs')
    parser.add_argument('--no-browser', action='store_true', help='Connect without opening a browser')
    args = parser.parse_args(argv)
    try:
        manifest, count = check_package()
        if args.check_package:
            print('Package verified: %d files plus the Pi bundle archive.' % count)
            return 0
        config = load_config()
        connection = Connection(ROOT, config)
        connection.connect()
        state = connection.prepare(manifest, install_if_missing=not args.download)
        if args.download:
            output = connection.download(state)
            if output:
                print('Downloads saved in ' + str(output))
        else:
            print('Flight lab is ready: ' + connection.url, flush=True)
            if not args.no_browser:
                webbrowser.open(connection.url)
        return 0
    except (LabError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print('Flight lab: ' + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nCanceled. No Pi process was stopped.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
