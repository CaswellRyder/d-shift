"""Bring up the persistent bench hotspot and its guarded WPA-PSK workaround."""

import argparse
import json
import subprocess
import time
import uuid

if __package__:
    from .pi_ap_compat import CompatError, run_compat
else:
    from pi_ap_compat import CompatError, run_compat


NMCLI = '/usr/bin/nmcli'
INTERFACE = 'wlan0'
SSID = 'DTR-Drone-Bench'


class BootError(Exception):
    """A sanitized startup failure; no successful AP state is implied."""


def bring_up(connection_uuid, *, runner=subprocess.run, compat=run_compat,
             clock=time.monotonic, sleep=time.sleep):
    """Activate one configured profile, then inspect or update its selected AP.

    The complete operation has an 80-second deadline. Compatibility mutation has
    an earlier deadline that reserves the final 20 seconds for the existing
    helper's guarded rollback. A failed or timed-out rollback remains a failure.
    """
    try:
        valid = str(uuid.UUID(connection_uuid)) == connection_uuid
    except (ValueError, AttributeError, TypeError):
        valid = False
    if not valid:
        raise BootError('A canonical connection UUID is required.')
    deadline = clock() + 80

    def bounded(args, *, limit=None, **kwargs):
        remaining = (deadline if limit is None else min(limit, deadline)) - clock()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(args, 0)
        kwargs['timeout'] = min(kwargs.get('timeout', remaining), remaining)
        return runner(args, **kwargs)

    for args, timeout in (
        ([NMCLI, '--wait', '20', 'radio', 'wifi', 'on'], 22),
        ([NMCLI, '--wait', '25', 'connection', 'up', 'uuid', connection_uuid,
          'ifname', INTERFACE], 27),
    ):
        try:
            result = bounded(args, capture_output=True, text=True, timeout=timeout,
                             check=False)
        except (OSError, subprocess.SubprocessError, UnicodeError):
            raise BootError('NetworkManager activation failed; AP state unverified.') from None
        if result.returncode:
            raise BootError('NetworkManager activation failed; AP state unverified.')

    inspection = None
    for attempt in range(5):
        try:
            inspection = compat(interface=INTERFACE, ssid=SSID, apply=False,
                                runner=bounded)
            break
        except CompatError:
            if attempt == 4 or clock() + 0.5 >= deadline - 20:
                raise BootError('Selected hotspot did not become ready; AP state unverified.') from None
            sleep(0.5)

    changed = False
    if inspection['configuration']['key_mgmt'] != 'WPA-PSK':
        mutation_deadline = deadline - 20
        if clock() >= mutation_deadline:
            raise BootError('Startup deadline reached before AP update; AP state unverified.')
        recovering = False

        def mutation_runner(args, **kwargs):
            nonlocal recovering
            try:
                result = bounded(args, limit=deadline if recovering else mutation_deadline,
                                 **kwargs)
            except (OSError, subprocess.SubprocessError, UnicodeError):
                recovering = True
                raise
            if result.returncode or result.stdout.rstrip('\r\n') in ('FAIL', 'UNKNOWN COMMAND'):
                recovering = True
            return result

        try:
            compat(interface=INTERFACE, ssid=SSID, apply=True, runner=mutation_runner)
        except CompatError:
            raise BootError('Compatibility update failed; AP state unverified.') from None
        changed = True
    if clock() >= deadline:
        raise BootError('Startup deadline reached; AP state unverified.')
    return dict(interface=INTERFACE, ssid=SSID, connection_uuid=connection_uuid,
                compatibility_changed=changed, client_connection_verified=False)


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        super().error('Invalid arguments; use --help for usage.')


def main(argv=None):
    parser = SafeParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--uuid', required=True, dest='connection_uuid')
    args = parser.parse_args(argv)
    try:
        result = bring_up(args.connection_uuid)
    except BootError as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
