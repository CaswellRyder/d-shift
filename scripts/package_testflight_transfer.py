"""Build an offline MacBook operator kit and a separate private pairing ZIP."""

import argparse
import base64
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).with_name('testflight_templates')
DEPENDENCIES = {'numpy': '1.26.4', 'opencv-python-headless': '4.11.0.86', 'pillow': '11.1.0'}
ARCHITECTURES = ('arm64', 'amd64')
SOURCE_TREES = {'src': {'.py', '.c', '.h', '.html', '.css', '.js'},
                'network': {'.go', '.mod', '.sum', '.js', '.mjs', '.html', '.css', '.sh',
                            '.example'},
                'firmware': {'.h', '.ino', '.cpp', '.c'},
                'tests': {'.py', '.js', '.mjs'}}
SCRIPT_NAMES = ('run_ground_station.py', 'process_raw_capture.py', 'pi_flight_lab.py',
                'pi_flight_loop.py', 'capture_calibration_frames.py', 'calibrate_lens.py',
                'pi_hotspot_profile.py', 'pi_ap_compat.py', 'lab_session.py',
                'pi_hotspot_boot.py',
                'package_testflight_transfer.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def regular(path, *, private=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Input must be a regular file, not a symbolic link')
    if private and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('Credential input must have private permissions')
    return path


def copy_file(source, destination, *, executable=False):
    regular(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o700 if executable else 0o600)


def validate_python_archive(path):
    """Validate all archive paths before the Mac shell extracts the trusted bytes."""
    regular(path)
    names, links, total, interpreter = set(), {}, 0, False
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) > 100_000:
            raise ValueError('Python archive has too many members')
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or not name.parts
                    or name.parts[0] != 'python' or '\\' in member.name
                    or member.name in names):
                raise ValueError('Unsafe or duplicate Python archive member')
            names.add(member.name)
            if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise ValueError('Unsupported Python archive member type')
            total += member.size
            if total > 1024 * 1024 * 1024:
                raise ValueError('Python archive exceeds the installed-size bound')
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or '\\' in member.linkname:
                    raise ValueError('Unsafe Python archive link')
                base = [] if member.islnk() else list(name.parent.parts)
                for part in target.parts:
                    if part == '..':
                        if len(base) <= 1:
                            raise ValueError('Python archive link escapes its root')
                        base.pop()
                    elif part not in ('.', ''):
                        base.append(part)
                if not base or base[0] != 'python':
                    raise ValueError('Python archive link escapes its root')
                links[name.as_posix()] = PurePosixPath(*base)
            if name.as_posix() == 'python/bin/python3.11':
                interpreter = member.isfile() or member.issym() or member.islnk()
        for member in members:
            if any(parent.as_posix() in links for parent in PurePosixPath(member.name).parents):
                raise ValueError('Python archive member is nested beneath a link')
        # Targets must resolve within the archive; reject link cycles and dangling links.
        for source, target in links.items():
            visited = {source}
            while target.as_posix() in links:
                if target.as_posix() in visited:
                    raise ValueError('Python archive has a link cycle')
                visited.add(target.as_posix())
                target = links[target.as_posix()]
            if target.as_posix() not in names:
                raise ValueError('Python archive has a dangling link')
            if any(parent.as_posix() in links for parent in target.parents):
                raise ValueError('Python archive link target traverses another link')
    if not interpreter:
        raise ValueError('Python archive must contain python/bin/python3.11')


def wheel_requirements(wheelhouse):
    wheels = sorted(Path(wheelhouse).glob('*.whl'))
    records = {}
    for wheel in wheels:
        regular(wheel)
        with zipfile.ZipFile(wheel) as archive:
            metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:
                raise ValueError('Wheel must have one package metadata record')
            message = BytesParser().parsebytes(archive.read(metadata[0]))
        name = re.sub(r'[-_.]+', '-', message['Name']).lower()
        version = message['Version']
        if name not in DEPENDENCIES or version != DEPENDENCIES[name] or name in records:
            raise ValueError('Wheelhouse must contain exactly the pinned review dependencies')
        records[name] = (wheel, digest(wheel))
    if set(records) != set(DEPENDENCIES):
        raise ValueError('Wheelhouse is missing a pinned review dependency')
    requirements = ''.join(f'{name}=={DEPENDENCIES[name]} --hash=sha256:{records[name][1]}\n'
                           for name in sorted(records))
    return requirements, [records[name][0] for name in sorted(records)]


def snapshot_sources(repo, folder):
    source = folder / 'source'
    copied = {}
    selected = []
    for dirname, suffixes in SOURCE_TREES.items():
        for path in sorted((repo / dirname).rglob('*')):
            if path.is_file() and path.suffix in suffixes and not any(
                    part in ('__pycache__', 'bin', '.runtime') for part in path.parts):
                selected.append(path)
    for name in SCRIPT_NAMES:
        path = repo / 'scripts' / name
        if path.is_file():
            selected.append(path)
    for name in ('camera_mount.json', 'target_sizes.json'):
        path = repo / 'configs' / name
        if path.is_file():
            selected.append(path)
    for name in ('README.md', 'pyproject.toml', 'uv.lock', 'LICENSE', 'THIRD_PARTY_NOTICES.txt',
                 'THIRD_PARTY_NOTICES.md'):
        path = repo / name
        if path.is_file():
            selected.append(path)
    for path in selected:
        relative = path.relative_to(repo).as_posix()
        destination = source / relative
        copy_file(path, destination)
        copied[relative] = digest(destination)
    required = ('scripts/run_ground_station.py', 'scripts/process_raw_capture.py',
                'src/dtr/raw_capture.py', 'network/go.mod')
    if any(name not in copied for name in required):
        raise ValueError('Current source is missing a required station component')
    protocol = (source / 'network/protocol/types.go').read_text()
    if not re.search(r'\bVersion\s*=\s*4\b', protocol):
        raise ValueError('This builder expects Go wire protocol v4')
    return copied


def build_binary(repo, destination, platform_name, architecture, command, go):
    env = dict(os.environ, CGO_ENABLED='0', GOOS=platform_name, GOARCH=architecture)
    if platform_name == 'linux' and architecture == 'arm':
        env['GOARM'] = '6'
    subprocess.run([go, 'build', '-trimpath', '-ldflags=-s -w', '-o', str(destination),
                    './cmd/' + command], cwd=repo / 'network', env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    regular(destination)
    if destination.stat().st_size == 0:
        raise ValueError('Compiler produced an empty executable')
    destination.chmod(0o700)


def credentials(key_file, ssh_identity, known_hosts, wifi_password_file):
    key_file = regular(key_file, private=True)
    ssh_identity = regular(ssh_identity, private=True)
    known_hosts = regular(known_hosts)
    wifi_password_file = regular(wifi_password_file, private=True)
    if key_file.stat().st_size != 32 or not ssh_identity.stat().st_size:
        raise ValueError('Pairing key must be 32 raw bytes and SSH identity must be nonempty')
    password_length = len(wifi_password_file.read_bytes().rstrip(b'\r\n'))
    if not 8 <= password_length <= 63:
        raise ValueError('Wi-Fi password file has an invalid WPA2 password length')
    host_lines = [line.split() for line in known_hosts.read_text().splitlines()
                  if line.strip() and not line.startswith('#')]
    if not host_lines or any(len(line) != 3 or line[0] != 'vision.local'
                             or line[1] not in ('ssh-ed25519', 'ecdsa-sha2-nistp256', 'ssh-rsa')
                             for line in host_lines):
        raise ValueError('Known-hosts input must contain verified vision.local public host keys only')
    for line in host_lines:
        base64.b64decode(line[2], validate=True)
    return {'ground.key': key_file, 'capture_ed25519': ssh_identity,
            'known_hosts': known_hosts, 'wifi-password.txt': wifi_password_file}


def archive_folder(folder, archive):
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for path in [folder, *sorted(folder.rglob('*'))]:
            if path.is_symlink():
                raise ValueError('Transfer folder unexpectedly contains a symbolic link')
            stream.write(path, path.relative_to(folder.parent).as_posix())
    archive.chmod(0o600)


README = '''DTR MacBook test-flight preparation kit

This is the current recording-only operator station, Go wire protocol v4.
It includes Apple silicon and Intel Mac programs, Python 3.11 runtimes, pinned
offline review dependencies, current working-tree source and supplied Pi recovery
assets. No Go, Homebrew, desktop Mac, internet connection or copied .venv is needed
for initial setup. macOS 13 or newer is required by the pinned review wheels;
first launch on the actual MacBook must be checked before departure.

Transfer BOTH ZIPs to the MacBook using a private local transfer such as AirDrop:
  PROGRAM.zip: application, offline dependencies, source and recovery assets.
  PRIVATE.zip: Pi pairing key, capture-only SSH identity, pinned SSH host key and
               hotspot password. Keep this ZIP/private folder private.
Compare both whole-ZIP SHA-256 values with the sender before trusting them.

1. Extract both ZIPs in the same writable local folder. Keep their folder names.
   Do not put the program inside the PRIVATE folder. Do not run from a ZIP viewer.
2. Open Verify Kit.command. Every program file must verify OK.
3. Open Setup MacBook.command. It imports credentials without replacing existing
   pairing files, extracts this Mac's Python and installs review wheels OFFLINE.
4. Power the Pi. Join DTR-Drone-Bench using the private wifi-password.txt file.
5. Open Start Ground Station.command and leave its Terminal open.
6. Open http://127.0.0.1:8782. Wait for fresh Pi bench telemetry. No command is sent
   automatically. Resume and Enable are separate, explicit actions when needed.
7. Record a diagnostic take, wait until complete, then Transfer & process and open
   its review. Captures and private logs stay in this kit's Captures/.runtime dirs.
8. Check reconnect, Pause, E-stop/reset and no old held-input replay. These checks
   establish Pi bench handling only; they do not establish motor behaviour.

Control-C stops this station and its own capture tunnel. No login job is installed.
The existing supervisor retries after Wi-Fi changes. The Wi-Fi device is detected
on this MacBook; desktop interface en1 and desktop absolute paths are not reused.
Keep the extracted runtime/site directories with the kit if moving it later.
The capture processor uses original Pi frames and decisions; it does not rerun
inference. A restricted SSH identity only forwards Pi loopback port 8771. Its
public key must already be authorized on the Pi. SSH uses the bundled pinned
vision.local host key, strict checking, no agent/default identities or global config.

The Pi hotspot, one camera/vision process and Pi bridge are prerequisites. This
kit does not start a camera, replace models, change calibration, or enable motors.
Current receipts are bench_only=true/applied=false. Launch, ESP32 application,
physical position hold, motor cutoff and flight qualification remain separate.
Read HANDOFF-STATUS.txt for the Pi-side acceptance boundary supplied by the sender.

If macOS blocks an unsigned transferred script/binary, verify the ZIP hash first,
then approve that specific item with Finder Open or Privacy & Security. Do not
disable Gatekeeper or clear quarantine globally. If another console/tunnel owns
port 8782 or 8771, stop it from its own Terminal; this kit will not take it over.
The bundled manifests detect changed bytes, not malicious replacement of an entire
bundle. SOURCE-MANIFEST.json labels this as a current uncommitted working-tree
snapshot; an older release or plain Git clone is not equivalent.

No recordings, datasets, training caches, shell/session history, credentials or
private keys are included in PROGRAM.zip. Additional recovery assets are included
only when explicitly provided to the builder and are listed in MANIFEST.json.
'''


def package(repo, output, *, python_archives, wheelhouses, key_file, ssh_identity,
            known_hosts, wifi_password_file, recovery_files=None, status_file=None,
            go='go', compiler=build_binary):
    repo, output = Path(repo).resolve(), Path(output).absolute()
    private_output = output.with_name(output.name + '-PRIVATE')
    program_zip = output.with_name(output.name + '-PROGRAM.zip')
    private_zip = output.with_name(output.name + '-PRIVATE.zip')
    destinations = (output, private_output, program_zip, private_zip)
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError('A transfer output already exists; nothing was overwritten')
    inputs = credentials(key_file, ssh_identity, known_hosts, wifi_password_file)
    for arch in ARCHITECTURES:
        validate_python_archive(python_archives[arch])
        wheel_requirements(wheelhouses[arch])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.testflight-transfer-', dir=output.parent))
    folder, private_folder = staging / output.name, staging / private_output.name
    owned = []
    try:
        folder.mkdir(mode=0o700)
        private_folder.mkdir(mode=0o700)
        snapshot = snapshot_sources(repo, folder)
        for arch in ARCHITECTURES:
            for command in ('laptop-link', 'linkctl'):
                target = folder / 'bin' / (command + '-' + arch)
                target.parent.mkdir(exist_ok=True)
                compiler(folder / 'source', target, 'darwin', arch, command, go)
                regular(target)
                target.chmod(0o700)
            offline = folder / 'offline'
            copy_file(python_archives[arch], offline / ('python-' + arch + '.tar.gz'))
            requirements, wheels = wheel_requirements(wheelhouses[arch])
            (offline / ('requirements-' + arch + '.txt')).write_text(requirements)
            for wheel in wheels:
                copy_file(wheel, offline / ('wheels-' + arch) / wheel.name)
        target = folder / 'recovery/pi-link-armv6'
        target.parent.mkdir(exist_ok=True)
        compiler(folder / 'source', target, 'linux', 'arm', 'pi-link', go)
        regular(target)
        target.chmod(0o700)
        for name, source in (recovery_files or {}).items():
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name):
                raise ValueError('Recovery name must be a plain safe filename')
            if (folder / 'recovery' / name).exists():
                raise ValueError('Recovery asset name collides with a generated file')
            copy_file(source, folder / 'recovery' / name)
        for name, source in inputs.items():
            copy_file(source, private_folder / 'private' / name)
        (private_folder / 'KEEP PRIVATE.txt').write_text(
            'This folder contains Pi credentials. Keep it private. Extract beside the program kit,\n'
            'then run Setup MacBook.command. Do not share with public releases or source control.\n')
        (folder / 'README.txt').write_text(README)
        for name, operation in (('Setup MacBook.command', 'setup'),
                                ('Start Ground Station.command', 'start')):
            path = folder / name
            path.write_text((TEMPLATES / 'entry.command').read_text().replace('@OPERATION@', operation))
            path.chmod(0o700)
        verify = folder / 'Verify Kit.command'
        verify.write_text('#!/bin/bash\nset -eu\ncd -P "$(dirname "$0")"\n'
                          '/usr/bin/shasum -a 256 -c SHA256SUMS\n'
                          'echo "Program files verified. PRIVATE pairing files are checked at setup."\n')
        verify.chmod(0o700)
        for name in ('macbook_station.py', 'capture-python'):
            copy_file(TEMPLATES / name, folder / 'tools' / name, executable=True)
        # Rebuild templates are part of the source snapshot as well.
        for name in ('macbook_station.py', 'capture-python', 'entry.command'):
            destination = folder / 'source/scripts/testflight_templates' / name
            copy_file(TEMPLATES / name, destination)
            snapshot[destination.relative_to(folder / 'source').as_posix()] = digest(destination)
        (folder / 'SOURCE-MANIFEST.json').write_text(json.dumps(
            dict(kind='uncommitted-working-tree-snapshot', snapshot_is_committed=False,
                 files=snapshot), indent=2) + '\n')
        if status_file is not None:
            copy_file(status_file, folder / 'HANDOFF-STATUS.txt')
        else:
            (folder / 'HANDOFF-STATUS.txt').write_text(
                'No Pi hardware acceptance status was supplied to this builder.\n'
                'MacBook setup and live Pi checks remain required. ESP32 execution, physical\n'
                'calibration, power-cycle startup and flight qualification are not implied.\n')
        # Every executable is compiled from the manifested snapshot, so later
        # additions, edits or deletions in the live checkout cannot change it.
        manifest = dict(schema='dtr.macbook-transfer.v1', wire_version=4, minimum_macos=13,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        includes_credentials=False, includes_recordings=False,
                        first_macbook_launch_verified=False, motor_execution_available=False,
                        files={p.relative_to(folder).as_posix(): digest(p)
                               for p in sorted(folder.rglob('*')) if p.is_file()})
        (folder / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (folder / 'SHA256SUMS').write_text(''.join(
            f'{digest(path)}  {path.relative_to(folder).as_posix()}\n'
            for path in sorted(folder.rglob('*')) if path.is_file()))
        for tree in (folder, private_folder):
            for path in tree.rglob('*'):
                path.chmod(0o700 if path.is_dir() or path.suffix == '.command'
                           or path.parent.name == 'bin' or path.name in ('capture-python', 'pi-link-armv6')
                           else 0o600)
        for tree, final in ((folder, program_zip), (private_folder, private_zip)):
            temporary = staging / final.name
            archive_folder(tree, temporary)
            os.link(temporary, final)
            owned.append(final)
        for tree, final in ((folder, output), (private_folder, private_output)):
            if final.exists() or final.is_symlink():
                raise FileExistsError('Transfer destination appeared while building')
            tree.rename(final)
            owned.append(final)
        return dict(program_folder=str(output), private_folder=str(private_output),
                    program_zip=str(program_zip), private_zip=str(private_zip),
                    program_zip_sha256=digest(program_zip), private_zip_sha256=digest(private_zip),
                    capture_review_included=True, credentials_separate=True,
                    macbook_live_check_complete=False)
    except BaseException:
        for path in reversed(owned):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        raise
    finally:
        shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(ROOT))
    parser.add_argument('--output', required=True, help='New program folder; never overwritten')
    for arch in ARCHITECTURES:
        parser.add_argument('--python-' + arch, required=True, help='Verified Python 3.11 install_only tar.gz')
        parser.add_argument('--wheelhouse-' + arch, required=True, help='Pinned offline wheels for this Mac arch')
    for name in ('key-file', 'ssh-identity', 'known-hosts', 'wifi-password-file'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--recovery-file', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--status-file', help='Current factual Pi acceptance handoff text')
    parser.add_argument('--go', default='go')
    args = parser.parse_args(argv)
    try:
        recovery = {}
        for item in args.recovery_file:
            name, separator, filename = item.partition('=')
            if not separator or name in recovery:
                raise ValueError('Recovery files require unique NAME=PATH arguments')
            recovery[name] = filename
        result = package(args.repo, args.output,
                         python_archives={arch: getattr(args, 'python_' + arch) for arch in ARCHITECTURES},
                         wheelhouses={arch: getattr(args, 'wheelhouse_' + arch) for arch in ARCHITECTURES},
                         key_file=args.key_file, ssh_identity=args.ssh_identity,
                         known_hosts=args.known_hosts, wifi_password_file=args.wifi_password_file,
                         recovery_files=recovery, status_file=args.status_file, go=args.go)
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile):
        parser.exit(1, 'Transfer build failed. Check pinned input assets, credentials, '
                    'toolchain and unused output paths. No credential contents were printed.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
