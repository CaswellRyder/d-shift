"""Generate a private NetworkManager Wi-Fi AP profile; never install or activate it."""

import argparse
import hashlib
import ipaddress
import os
from pathlib import Path
import re
import uuid


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's normal errors can repeat unknown arguments containing secrets.
        super().error("Invalid arguments; use --help for usage.")


def write_profile(password_file, output, *, ssid="DTR-Drone", interface="wlan0",
                  address="192.168.50.1/24", name="dtr-drone-link", disable_pmf=False):
    """Create an exclusive mode-0600 keyfile, returning its new connection UUID."""
    if type(disable_pmf) is not bool:
        raise ValueError("disable_pmf must be a boolean.")
    if (not re.fullmatch(r"[A-Za-z0-9 _-]{1,32}", ssid)
            or ssid != ssid.strip()):
        raise ValueError("SSID needs 1..32 ASCII letters, digits, spaces, underscores or dashes; "
                         "no leading or trailing spaces.")
    for value, limit, label in ((interface, 15, "Interface"), (name, 64, "Connection name")):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0," + str(limit - 1) + r"}", value):
            raise ValueError(f"{label} needs a safe alphanumeric, underscore or dash identifier.")
    try:
        host = ipaddress.IPv4Interface(address)
    except ValueError:
        raise ValueError("Address needs a private IPv4 host with a /24 prefix.") from None
    private = any(host.ip in ipaddress.IPv4Network(block)
                  for block in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
    if (not private or host.network.prefixlen != 24
            or host.ip in (host.network.network_address, host.network.broadcast_address)):
        raise ValueError("Address needs a private IPv4 host with a /24 prefix.")
    output = Path(output)
    if output.suffix != ".nmconnection":
        raise ValueError("Output needs the .nmconnection suffix.")
    with Path(password_file).open("rb") as stream:
        password = stream.read(65)
    if password.endswith(b"\n"):
        password = password[:-1]  # Permit one terminal LF; preserve passphrase spaces.
    if not 8 <= len(password) <= 63 or any(byte < 32 or byte > 126 for byte in password):
        raise ValueError("Password file needs 8..63 printable ASCII bytes and optional final LF.")
    psk = hashlib.pbkdf2_hmac("sha1", password, ssid.encode("ascii"), 4096, 32).hex()
    connection_uuid = str(uuid.uuid4())
    pmf = "pmf=1\n" if disable_pmf else ""
    content = (f"[connection]\nid={name}\nuuid={connection_uuid}\ntype=wifi\n"
               f"interface-name={interface}\nautoconnect=true\n\n"
               f"[wifi]\nmode=ap\nband=bg\nchannel=6\nssid={ssid}\n\n"
               f"[wifi-security]\n{pmf}key-mgmt=wpa-psk\nproto=rsn\npairwise=ccmp\n"
               f"group=ccmp\npsk={psk}\n\n"
               f"[ipv4]\nmethod=shared\naddress1={host}\nnever-default=true\n\n"
               "[ipv6]\nmethod=disabled\n")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as stream:
        os.fchmod(stream.fileno(), 0o600)  # Keep exact private mode with restrictive umasks too.
        stream.write(content)
    return connection_uuid


def main(argv=None):
    parser = SafeArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--password-file", required=True, help="Private file holding the passphrase")
    parser.add_argument("--output", required=True, help="New .nmconnection path; never overwritten")
    parser.add_argument("--ssid", default="DTR-Drone")
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--address", default="192.168.50.1/24")
    parser.add_argument("--name", default="dtr-drone-link")
    parser.add_argument("--disable-pmf", action="store_true",
                        help="disable 802.11w for older AP drivers; WPA2/CCMP stays enabled")
    args = parser.parse_args(argv)
    try:
        connection_uuid = write_profile(**vars(args))
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    except FileExistsError:
        parser.exit(2, f"{parser.prog}: error: Output already exists; choose a new path.\n")
    except OSError:
        parser.exit(2, f"{parser.prog}: error: Could not read password or create private output.\n")
    print(f"Created {args.output}: name={args.name}, SSID={args.ssid}, UUID={connection_uuid}")
    print("Profile contains a network secret. Keep it private (mode 0600).")


if __name__ == "__main__":
    main()
