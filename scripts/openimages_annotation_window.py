"""Parse a bounded interior TRAIN CSV window without admitting clipped image groups."""
import csv
import io
import re

from scripts.acquire_openimages_balloons import BALLOON


def complete_window(raw, header):
    if not isinstance(raw, bytes) or not isinstance(header, bytes) or b'\n' in header.rstrip(b'\r\n'):
        raise ValueError('Require raw window and one CSV header')
    fields = next(csv.reader([header.decode().strip()]))
    required = {'ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax'}
    if not required <= set(fields) or len(fields) != len(set(fields)):
        raise ValueError('Unexpected annotation header')
    first_newline, last_newline = raw.find(b'\n'), raw.rfind(b'\n')
    if first_newline < 0 or last_newline <= first_newline:
        return {}
    # Conservatively skip the first line even if the window begins at a boundary.
    body = raw[first_newline+1:last_newline+1].decode('utf-8')
    groups, first, previous = {}, None, None
    for row in csv.DictReader(io.StringIO(body), fieldnames=fields):
        identifier = row['ImageID']
        if (None in row or any(row[k] is None for k in required)
                or not re.fullmatch('[0-9a-f]{16}', identifier)
                or (previous is not None and identifier < previous)):
            raise ValueError('Incomplete, invalid or unsorted annotation group')
        first = identifier if first is None else first
        previous = identifier
        if row['LabelName'] != BALLOON:
            continue
        box = [float(row[k]) for k in ('XMin', 'YMin', 'XMax', 'YMax')]
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError('Invalid source box')
        groups.setdefault(identifier, []).append(row)
    # The first/last observed groups may extend outside the downloaded window.
    groups.pop(first, None)
    groups.pop(previous, None)
    return groups
