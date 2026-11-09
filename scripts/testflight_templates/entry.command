#!/bin/bash
set -eu
umask 077
kit_dir="$(cd -P "$(dirname "$0")" && pwd)"
cd "$kit_dir"
if [ "$(/usr/bin/uname -s)" != Darwin ]; then
  echo 'This transfer kit requires macOS.' >&2; exit 1
fi
macos_version="$(/usr/bin/sw_vers -productVersion)"
macos_major="${macos_version%%.*}"
if ! [[ "$macos_major" =~ ^[0-9]+$ ]] || [ "$macos_major" -lt 13 ]; then
  echo 'This offline capture kit requires macOS 13 or newer.' >&2; exit 1
fi
case "$(/usr/bin/uname -m)" in
  arm64) architecture=arm64 ;;
  x86_64) architecture=amd64 ;;
  *) echo 'Unsupported Mac processor.' >&2; exit 1 ;;
esac
/usr/bin/shasum -a 256 -c SHA256SUMS >/dev/null || {
  echo 'Kit checksum verification failed. Copy the complete original kit.' >&2; exit 1;
}
if [ -L .runtime ]; then echo 'Runtime directory cannot be a symbolic link.' >&2; exit 1; fi
mkdir -p .runtime
chmod 700 .runtime
runtime="$kit_dir/.runtime/python-$architecture"
if [ -L "$runtime" ]; then echo 'Python runtime cannot be a symbolic link.' >&2; exit 1; fi
if [ ! -d "$runtime" ]; then
  pending="$(mktemp -d "$kit_dir/.runtime/python-pending-XXXXXXXX")"
  trap 'rm -rf "$pending"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  /usr/bin/tar -xzf "$kit_dir/offline/python-$architecture.tar.gz" -C "$pending"
  if [ ! -x "$pending/python/bin/python3.11" ]; then
    echo 'Bundled Python archive did not contain its required interpreter.' >&2; exit 1
  fi
  mv "$pending" "$runtime"
  trap - EXIT INT TERM HUP
fi
exec "$runtime/python/bin/python3.11" -I "$kit_dir/tools/macbook_station.py" @OPERATION@
