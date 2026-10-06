#!/bin/sh
set -eu
: "${DTR_ZIG:?Set DTR_ZIG to the pinned Zig executable}"
exec "$DTR_ZIG" cc -target arm-linux-gnueabihf.2.28 -mcpu=arm1176jzf_s "$@"
