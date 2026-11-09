"""Inspect a bench AP; explicitly apply a temporary WPA-PSK compatibility rebuild."""

import argparse
import json
import re
import subprocess

WPA_CLI = "/usr/sbin/wpa_cli"
FIELDS = ("mode", "proto", "pairwise", "group", "ieee80211w", "key_mgmt")


class CompatError(Exception):
    """A sanitized control failure."""


class CommandRejected(CompatError):
    """An explicit control rejection, rather than an uncertain execution."""


def run_compat(*, interface="wlan0", ssid="DTR-Drone-Bench", apply=False,
               runner=subprocess.run):
    """Inspect or rebuild only the unique current, security-validated bench AP.

    The injected runner follows subprocess.run's contract. No secret fields are read.
    This changes volatile supplicant configuration only; NM may replace it later.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,14}", interface):
        raise CompatError("Invalid interface identifier.")
    if not re.fullmatch(r"[A-Za-z0-9 _-]{1,32}", ssid) or ssid != ssid.strip():
        raise CompatError("Invalid ASCII SSID.")

    def command(*args):
        try:
            result = runner([WPA_CLI, "-i", interface, *args], capture_output=True,
                            text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError, UnicodeError):
            raise CompatError("wpa_cli execution failed.") from None
        value = result.stdout.rstrip("\r\n")
        if result.returncode:
            raise CompatError("wpa_cli command failed.")
        if value in ("FAIL", "UNKNOWN COMMAND"):
            raise CommandRejected("wpa_cli command rejected.")
        return value

    def inspect(require_current, expected_id=None):
        rows = command("list_networks").splitlines()
        if not rows or rows[0] != "network id / ssid / bssid / flags":
            raise CompatError("Unexpected network list format.")
        if len(rows) != 2:
            raise CompatError("A dedicated AP interface with exactly one network is required.")
        matches = []
        for row in rows[1:]:
            parts = row.split("\t")
            if len(parts) != 4 or not parts[0].isascii() or not parts[0].isdigit():
                raise CompatError("Unexpected network row format.")
            if parts[1] == ssid:
                matches.append(parts)
        if len(matches) != 1:
            raise CompatError("SSID must identify exactly one network.")
        network_id, _, _, flags = matches[0]
        if expected_id is not None and network_id != expected_id:
            raise CompatError("AP network identity changed.")
        if require_current and "[CURRENT]" not in flags:
            raise CompatError("Selected AP must be current before applying or inspecting.")
        values = {field: command("get_network", network_id, field) for field in FIELDS}
        expected = {"mode": "2", "proto": "RSN", "pairwise": "CCMP", "group": "CCMP",
                    "ieee80211w": "0"}
        if any(values[key] != value for key, value in expected.items()):
            raise CompatError("Selected network is not the required WPA2/CCMP AP with PMF disabled.")
        key_mgmt = values["key_mgmt"].split()
        if (not key_mgmt or "WPA-PSK" not in key_mgmt
                or not set(key_mgmt) <= {"WPA-PSK", "WPA-PSK-SHA256"}):
            raise CompatError("Selected AP has unsupported key management.")
        return network_id, flags, values

    network_id, flags, before = inspect(True)
    mutation_attempted = False
    change_acknowledged = False

    def write(*args):
        nonlocal mutation_attempted
        inspect(False, expected_id=network_id)
        mutation_attempted = True
        if command(*args) != "OK":
            raise CompatError("wpa_cli did not acknowledge change.")

    if apply:
        try:
            write("set_network", network_id, "key_mgmt", "WPA-PSK")
            change_acknowledged = True
            for action in ("disable_network", "enable_network", "select_network"):
                write(action, network_id)
            after_id, flags, after = inspect(False, expected_id=network_id)
            if after_id != network_id or after["key_mgmt"] != "WPA-PSK":
                raise CompatError("Post-apply configuration verification failed.")
        except CompatError as error:
            if (not mutation_attempted
                    or (isinstance(error, CommandRejected) and not change_acknowledged)):
                raise CompatError("Apply rejected before any acknowledged change; no rollback needed.") from None
            try:
                inspect(False, expected_id=network_id)
            except CompatError:
                raise CompatError("Apply failed; rollback skipped because AP identity could not be validated.") from None
            outcomes = []
            for args in (("set_network", network_id, "key_mgmt", before["key_mgmt"]),
                         ("disable_network", network_id), ("enable_network", network_id),
                         ("select_network", network_id)):
                try:
                    write(*args)
                    outcomes.append(True)
                except CompatError:
                    outcomes.append(False)
            outcome = "commands acknowledged" if all(outcomes) else "one or more commands failed"
            raise CompatError(f"Apply failed; rollback {outcome}. Client connection unverified.") from None
    else:
        after = before
    return {"interface": interface, "ssid": ssid, "network_id": network_id,
            "current": "[CURRENT]" in flags, "configuration": after,
            "applied_config_only": bool(apply), "client_connection_verified": False,
            "persistent": False}


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        super().error("Invalid arguments; use --help for usage.")


def main(argv=None):
    parser = SafeParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--apply", action="store_true", help="Rebuild the validated current AP")
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--ssid", default="DTR-Drone-Bench")
    args = parser.parse_args(argv)
    try:
        result = run_compat(**vars(args))
    except CompatError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
