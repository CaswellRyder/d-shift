#!/bin/sh
set -eu
: "${DTR_ZIG:?Set DTR_ZIG to the pinned Zig executable}"
exec "$DTR_ZIG" ar "$@"
