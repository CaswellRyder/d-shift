"""Read a bounded preview from an existing loopback Pi viewer; GET requests only.

Use an explicit SSH local forward. Does not start/stop recording, change policy,
open the camera independently, or send any ESP32 commands. Preview requests make
the running viewer encode JPEGs briefly, so timing is not an isolated benchmark.
"""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from dtr.data import sha256, write_json

CONFIG_FIELDS = ('source', 'model', 'model_sha256', 'bundle_sha256', 'region_library_sha256',
                 'viewer_sha256', 'page_sha256', 'threshold', 'classes', 'max_golden_score_delta', 'golden_frames')


def preview_bytes(state):
    if state.get('error'):
        raise ValueError('Viewer reported an error: ' + str(state['error']))
    if not state.get('frame'):
        return None
    raw = base64.b64decode(state['frame'], validate=True)
    if len(raw) > 2_000_000:
        raise ValueError('Unexpected preview size')
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'JPEG' or image.size != (320, 240):
            raise ValueError('Unexpected preview geometry or encoding')
        image.verify()
    return raw


def sample(port, count, output):
    if not 1 <= count <= 10 or not 1024 <= port <= 65535:
        raise ValueError('Require bounded count and unprivileged loopback port')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = f'http://127.0.0.1:{port}'

    def get(path):
        with urlopen(base + path, timeout=5) as response:
            raw = response.read(3_000_001)
        if len(raw) > 3_000_000:
            raise ValueError('Response too large')
        return json.loads(raw)

    config = get('/api/config')
    if config.get('source') != 'camera':
        raise ValueError('Require actual camera source, not replay')
    # Keep an explicit allowlist: never write the viewer control token.
    info = {key: config[key] for key in CONFIG_FIELDS if key in config}
    output.mkdir(parents=True)
    rows, after, previous_sensor = [], 0, None
    for _ in range(count + 4):
        state = get(f'/api/state?after={after}')
        raw = preview_bytes(state)
        sequence = state.get('sequence', 0)
        if sequence <= after:
            raise ValueError('Viewer did not publish a fresh sequence')
        after = sequence
        if raw is None:
            continue  # First GET activates preview encoding for subsequent frames.
        row = state.get('row', {})
        sensor = row.get('sensor_timestamp_ns')
        if (not row.get('timestamp_valid') or not isinstance(sensor, int)
                or (previous_sensor is not None and sensor <= previous_sensor)
                or not 0 <= state.get('age_ms', float('inf')) <= 2000):
            raise ValueError('Preview timestamp is invalid, stale or repeated')
        previous_sensor = sensor
        path = output / f'{len(rows):03d}.jpg'
        path.write_bytes(raw)
        rows.append(dict(path=path.name, sha256=sha256(path), sequence=sequence,
                         row=row, age_ms=state['age_ms'], policy=state.get('policy'),
                         stats=state.get('stats'), health=state.get('health'), recording=state.get('recording')))
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError('Insufficient fresh preview frames')
    report = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), config=info, frames=rows,
        script_sha256=sha256(__file__), requested_frames=count,
        scope='Existing live preview only; JPEG frames are not raw training recordings. No ground truth, independent accuracy or isolated speed measurement.',
        control_requests_sent=False, test_evaluated=False, qualification_achieved=False)
    write_json(output / 'preview.json', report)
    print(json.dumps(dict(path=str(output), frames=len(rows), model=info['model'],
        model_sha256=info['model_sha256'], health=rows[-1]['health'],
        last_live_stats=rows[-1]['stats'], receipt_sha256=sha256(output / 'preview.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sample(args.port, args.count, args.output)
