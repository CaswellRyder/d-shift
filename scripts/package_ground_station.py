"""Build a private, portable macOS bench-console kit for a separate laptop."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile


TOOLS = {
    'uname': '/usr/bin/uname', 'networksetup': '/usr/sbin/networksetup',
    'awk': '/usr/bin/awk', 'curl': '/usr/bin/curl', 'lsof': '/usr/sbin/lsof',
    'open': '/usr/bin/open', 'sleep': '/bin/sleep', 'shasum': '/usr/bin/shasum',
    'stat': '/usr/bin/stat',
}


def launcher_script(tools=None):
    """Render fixed tool paths; an injected map permits network-free shell tests."""
    replacements = TOOLS if tools is None else dict(TOOLS, **tools)
    script = r'''#!/bin/bash
set -eu
umask 077
kit_dir="$(cd -P "$(dirname "$0")" && pwd)"
cd "$kit_dir"
if [ "$(@uname@ -s)" != Darwin ]; then
  echo 'This kit requires macOS.' >&2; exit 1
fi
case "$(@uname@ -m)" in
  arm64) binary="$kit_dir/bin/laptop-link-arm64" ;;
  x86_64) binary="$kit_dir/bin/laptop-link-amd64" ;;
  *) echo 'Unsupported Mac architecture.' >&2; exit 1 ;;
esac
@shasum@ -a 256 -c SHA256SUMS >/dev/null || {
  echo 'Kit verification failed. Copy a fresh complete kit.' >&2; exit 1;
}
key_file="$kit_dir/private/ground.key"
if [ -L "$key_file" ] || [ ! -f "$key_file" ] || [ "$(@stat@ -f '%z' "$key_file")" != 32 ]; then
  echo 'The private pairing key must be a regular 32-byte file.' >&2; exit 1
fi
chmod 700 "$kit_dir" "$kit_dir/private"
chmod 600 "$key_file"
wifi_interface="$(@networksetup@ -listallhardwareports | @awk@ '
  /^Hardware Port: (Wi-Fi|AirPort)$/ {wifi=1; next}
  wifi && /^Device: / {print $2; exit}
  /^Hardware Port:/ {wifi=0}
')"
if ! [[ "$wifi_interface" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]]; then
  echo 'Could not identify the Mac Wi-Fi interface.' >&2; exit 1
fi
runtime="$kit_dir/.runtime"
if [ -L "$runtime" ]; then echo 'Runtime directory must not be a symbolic link.' >&2; exit 1; fi
mkdir -p "$runtime"
chmod 700 "$runtime"
lock_dir="$runtime/station.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo 'This kit is already running, or a prior forced shutdown left its lock.' >&2
  echo 'Stop the existing station Terminal with Control-C. See README.txt for recovery.' >&2
  exit 1
fi
station_pid=''
printf '%s\n' "$$" > "$lock_dir/pid"
child_running() {
  [ -n "$station_pid" ] && jobs -pr | @awk@ -v wanted="$station_pid" '$1==wanted {found=1} END {exit !found}'
}
cleanup() {
  status=$?
  trap - EXIT INT TERM HUP
  if child_running; then
    kill -TERM "$station_pid" 2>/dev/null || true
    remaining=5
    while child_running && [ "$remaining" -gt 0 ]; do
      @sleep@ 1; remaining=$((remaining-1))
    done
    if child_running; then kill -KILL "$station_pid" 2>/dev/null || true; fi
  fi
  if [ -n "$station_pid" ]; then wait "$station_pid" 2>/dev/null || true; fi
  rm -f "$lock_dir/pid"
  rmdir "$lock_dir" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if @lsof@ -nP -iTCP:8782 -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo 'Port 8782 already belongs to another service. Stop that station first.' >&2
  exit 1
fi
echo 'Join DTR-Drone-Bench Wi-Fi. This Terminal runs your laptop console.'
echo 'Manual bench controls: Launch is unavailable. Inspect Controller telemetry for ESP32 status.'
echo 'Captures are not configured in this basic kit. Leave this Terminal open; Control-C stops it.'
opened=0
delay=1
while :; do
  session="$(mktemp -d "$runtime/session-XXXXXXXX")"
  "$binary" --peer 192.168.50.1:47110 --interface "$wifi_interface" \
    --key-file "$key_file" --listen 127.0.0.1:8782 \
    --journal "$session/commands.jsonl" --seconds 0 >"$session/console.log" 2>&1 &
  station_pid=$!
  ready=0
  while child_running; do
    if @lsof@ -a -p "$station_pid" -nP -iTCP:8782 -sTCP:LISTEN -t >/dev/null 2>&1 && \
       @curl@ --noproxy '*' --connect-timeout 1 --max-time 2 --fail --silent \
         http://127.0.0.1:8782/api/state >"$session/last-state.json"; then
      ready=1
      echo 'Console ready: http://127.0.0.1:8782'
      if [ "$opened" -eq 0 ]; then
        @open@ http://127.0.0.1:8782 || echo 'Open the address above in your browser.'
        opened=1
      fi
      break
    fi
    @sleep@ 0.2
  done
  if wait "$station_pid"; then result=0; else result=$?; fi
  station_pid=''
  if [ "$ready" -eq 1 ]; then delay=1; fi
  echo "Station exited (status $result). Rejoining Wi-Fi retries in $delay seconds."
  echo "Attempt log: $session/console.log"
  @sleep@ "$delay"
  delay=$((delay*2))
  if [ "$delay" -gt 15 ]; then delay=15; fi
done
'''
    for name, path in replacements.items():
        script = script.replace('@' + name + '@', "'" + str(path).replace("'", "'\\''") + "'")
    return script


VERIFY_SCRIPT = '''#!/bin/bash
set -eu
cd -P "$(dirname "$0")"
/usr/bin/shasum -a 256 -c SHA256SUMS
echo 'Program files match this kit manifest. The private pairing key is checked at startup.'
echo 'Compare the PRIVATE.zip SHA-256 with the sender before trusting a transferred copy.'
'''


README = '''DTR private Mac flight-laptop console

This folder contains the pairing key for your Pi. Keep the folder and PRIVATE.zip
private; do not upload them to GitHub, public sharing links or a release.

1. Copy and extract the complete kit onto the Mac laptop you are taking.
2. Double-click Verify Kit.command. Program files must all verify OK.
3. Power the Pi and join DTR-Drone-Bench Wi-Fi using its separately supplied password.
4. Double-click Start Ground Station.command. Leave its Terminal window open.
5. The browser opens http://127.0.0.1:8782 only after this station owns its server.
   If it does not open automatically, enter that address yourself.
6. Wait for Pi bench link connected. If paused, select Resume, then Enable controls.
   Hold a direction to send bench input; release to clear it. Pause and E-stop are
   explicit actions. Closing the browser does not stop the station Terminal.

Basic controls need no Go, Python, Homebrew, repository checkout or desktop Mac.
Both Apple silicon and Intel executables are included. Wi-Fi is detected on this
laptop; en1 is not assumed. A Pi boot does not automatically open a laptop browser.
Start Ground Station.command starts the laptop service for this session; this kit
does not install a login job. It retries startup/reconnect after Wi-Fi changes.
Control-C or closing this kit's Terminal stops only its own station process.

Current command receipts confirm Pi bench handling only. No ESP32 flight-control
receiver/launch command/controller acknowledgement is integrated. Launch stays
unavailable; these controls do not establish physical hover or emergency cutoff.
Do not infer flight readiness from a connected page.

Capture transfer and review processing are NOT configured in this basic kit.
The dashboard displays Capture service not configured. Enabling those functions
needs the separate laptop Python environment, processor and SSH capture tunnel.
The Pi camera lab and hotspot must already be running; this launcher does not
start cameras, create Wi-Fi networks or send automatic flight commands.

Stop another localhost:8782 station before using this kit. A second copy will not
take over or kill it. If a forced shutdown left a lock, first confirm no station
is running in another Terminal and that http://127.0.0.1:8782 does not load. Rename
this kit's .runtime folder to .runtime-old to preserve logs, then retry.
Do not remove a live station's lock. Logs and journals remain inside .runtime.

If macOS blocks a transferred script or executable, inspect this kit and its
checksum first, then use Finder's Open / System Settings > Privacy & Security to
approve this specific item. Do not disable Gatekeeper or clear quarantine globally.
If the Go executable is blocked, approve the matching file inside bin and try the
launcher again. Unsigned build availability and first launch on the actual laptop
still require verification.

Verification detects accidental file changes; a manifest carried in the same
folder is not a trusted signature. Compare the whole PRIVATE.zip hash supplied by
the sender. The pairing key is deliberately excluded from the program manifest.
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_binary(repo, destination, architecture, go):
    env = dict(os.environ, CGO_ENABLED='0', GOOS='darwin', GOARCH=architecture)
    subprocess.run([go, 'build', '-trimpath', '-ldflags=-s -w', '-o', str(destination),
                    './cmd/laptop-link'], cwd=repo / 'network', env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def package(repo, key_file, output, *, go='go', compiler=build_binary):
    repo, output, key_file = Path(repo).resolve(), Path(output).absolute(), Path(key_file)
    archive = output.with_name(output.name + '-PRIVATE.zip')
    for path in (output, archive):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Output folder or private ZIP already exists; nothing was overwritten')
    if key_file.is_symlink() or not key_file.is_file():
        raise ValueError('Pairing key must be a regular file')
    info = key_file.stat()
    if info.st_size != 32 or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError('Pairing key must contain 32 raw bytes with private permissions')
    key = key_file.read_bytes()
    if len(key) != 32:
        raise ValueError('Pairing key changed while being read')
    if not (repo / 'network/go.mod').is_file():
        raise ValueError('Repository is missing the Go network module')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.ground-station-', dir=output.parent))
    temp_zip = staging / 'private-transfer.zip'
    folder = staging / output.name
    folder.mkdir(mode=0o700)
    published = False
    archive_owned = False
    output_owned = False
    try:
        (folder / 'bin').mkdir(mode=0o700)
        (folder / 'private').mkdir(mode=0o700)
        for architecture in ('arm64', 'amd64'):
            target = folder / 'bin' / ('laptop-link-' + architecture)
            compiler(repo, target, architecture, go)
            if target.is_symlink() or not target.is_file() or target.stat().st_size == 0:
                raise ValueError('Compiler did not produce a regular executable')
            target.chmod(0o700)
        private_key = folder / 'private/ground.key'
        private_key.write_bytes(key)
        private_key.chmod(0o600)
        for name, content in [('Start Ground Station.command', launcher_script()),
                              ('Verify Kit.command', VERIFY_SCRIPT), ('README.txt', README)]:
            path = folder / name
            path.write_text(content)
            path.chmod(0o700 if path.suffix == '.command' else 0o600)
        files = {path.relative_to(folder).as_posix(): digest(path)
                 for path in sorted(folder.rglob('*'))
                 if path.is_file() and path != private_key}
        (folder / 'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name, value in files.items()))
        (folder / 'SHA256SUMS').chmod(0o600)
        with zipfile.ZipFile(temp_zip, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
            for path in [folder, *sorted(folder.rglob('*'))]:
                stream.write(path, path.relative_to(staging).as_posix())
        temp_zip.chmod(0o600)
        zip_hash = digest(temp_zip)
        # Reserve the output with mkdir rather than a replacing rename: even a
        # directory created by another build during compilation stays untouched.
        output.mkdir(mode=0o700)
        output_owned = True
        for child in folder.iterdir():
            child.rename(output / child.name)
        # The same-directory hard link also refuses a concurrently created ZIP.
        os.link(temp_zip, archive)
        archive_owned = True
        published = True
        return dict(folder=str(output), zip=str(archive), zip_sha256=zip_hash,
                    private=True, includes_pairing_key=True,
                    capture_service_configured=False, motor_execution_available=False)
    finally:
        shutil.rmtree(staging)
        if archive_owned and not published:
            archive.unlink()
        if output_owned and not published:
            shutil.rmtree(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--key-file', required=True, help='Existing private raw 32-byte Pi pairing key')
    parser.add_argument('--output', required=True, help='New private kit folder; never overwritten')
    parser.add_argument('--go', default='go', help='Go 1.25+ executable used only when building')
    args = parser.parse_args(argv)
    try:
        result = package(args.repo, args.key_file, args.output, go=args.go)
    except (OSError, ValueError, subprocess.SubprocessError):
        parser.exit(1, 'Private kit build failed; check inputs, Go toolchain and unused output paths.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
