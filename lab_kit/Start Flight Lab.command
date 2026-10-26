#!/bin/zsh
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ -x /usr/bin/python3 ]]; then
  LAB_PYTHON=/usr/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  LAB_PYTHON="$(command -v python3)"
else
  print -u2 'Python 3.9 or newer is required on the laptop. No pip packages are needed.'
  read -r '?Press Return to close. '
  exit 1
fi
"$LAB_PYTHON" "$SCRIPT_DIR/Laptop/flight_lab.py"
lab_status=$?
if [[ "$lab_status" -ne 0 ]]; then
  read -r '?Press Return to close. '
fi
exit "$lab_status"
