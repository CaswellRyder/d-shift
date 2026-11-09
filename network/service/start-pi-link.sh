#!/bin/sh
# Install alongside pi-link and ground.key in the private service directory.
set -eu
umask 077
cd /home/pacman/dtr-ground-station
mkdir -p journals
session_dir=$(mktemp -d journals/session-XXXXXXXX)
exec ./pi-link --bind 192.168.50.1 --port 47110 \
  --key-file /home/pacman/dtr-ground-station/ground.key \
  --journal "$session_dir/pi.jsonl" --seconds 0 --start-paused=true \
  --portal-listen 192.168.50.1:8782 \
  --source-url http://127.0.0.1:8771/api/telemetry
