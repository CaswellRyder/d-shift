#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
cmp "$repo/firmware/esp32_command_contract/command_contract.h" "$repo/firmware/esp32_manual_motors/command_contract.h"
"${CXX:-c++}" -std=c++11 -Wall -Wextra -Werror "$repo/firmware/esp32_manual_motors/test/controller_test.cpp" -o "$work/controller-test"
"$work/controller-test"
