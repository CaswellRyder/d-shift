"""Bounded, lossless capture transport; image processing belongs on the laptop.

v1 tar contains manifest.json and frames/NNNN.rgb regular files, no compression.
The manifest preserves original rows, events, identities and monotonic timestamps.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tarfile
import tempfile
import zipfile

import numpy as np

from dtr.flight_lab import CaptureBuffer, export_capture

SCHEMA = 'dtr.raw-capture.v1'
ID = re.compile(r'\d{8}T\d{6}-[0-9a-f]{8}')
MAX_MANIFEST = 8 * 1024 * 1024
MAX_RGB = 40 * 1024 * 1024
MAX_ARCHIVE = MAX_RGB + MAX_MANIFEST + 1024 * 1024


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def spool_capture(capture, root, identities, *, publish=True):
    if capture.state != 'exporting' or not capture.frames or not ID.fullmatch(capture.id):
        raise ValueError('Only finished captures can be spooled')
    folder = Path(root) / capture.id
    folder.mkdir(parents=True)
    pending = folder / 'capture.tar.partial'
    manifest = dict(schema=SCHEMA, id=capture.id, metadata=capture.metadata,
                    identities=identities, started=capture.started, finished_at=capture.finished_at,
                    duration_s=capture.duration_s, events=capture.events, frames=[])
    for index, (rgb, row) in enumerate(capture.frames):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected RGB uint8 frames')
        raw = rgb.tobytes()
        manifest['frames'].append(dict(path=f'frames/{index:04d}.rgb', shape=list(rgb.shape),
                                      bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), row=row))
    encoded = json.dumps(manifest, allow_nan=False).encode()
    if len(encoded) > MAX_MANIFEST or sum(f['bytes'] for f in manifest['frames']) > MAX_RGB:
        raise ValueError('Raw capture exceeds bounds')
    with tarfile.open(pending, 'w') as archive:
        member = tarfile.TarInfo('manifest.json')
        member.size = len(encoded)
        archive.addfile(member, io.BytesIO(encoded))
        for frame, (rgb, _) in zip(manifest['frames'], capture.frames):
            member = tarfile.TarInfo(frame['path'])
            member.size = frame['bytes']
            archive.addfile(member, io.BytesIO(rgb.tobytes()))
    final = folder / 'capture.tar'
    pending.rename(final)
    result = dict(id=capture.id, bytes=final.stat().st_size, sha256=digest(final),
                  frames=len(capture.frames), created_at=capture.metadata['created_at'],
                  source=identities['source'])
    (folder / 'raw-summary.json').write_text(json.dumps(result, allow_nan=False))
    if publish:
        capture.result, capture.state, capture.frames = result, 'complete', []
    return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_capture(path):
    """Read known regular members directly; never extract archive paths."""
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE:
        raise ValueError('Archive exceeds size bound')
    with tarfile.open(path, 'r:') as archive:
        members = []
        for member in archive:
            if len(members) >= 141 or not member.isfile() or member.size < 0:
                raise ValueError('Unexpected archive member type/count')
            members.append(member)
        if not members or members[0].name != 'manifest.json' or members[0].size > MAX_MANIFEST:
            raise ValueError('Missing or oversized manifest')
        manifest = json.loads(archive.extractfile(members[0]).read(),
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        if manifest.get('schema') != SCHEMA or not ID.fullmatch(manifest.get('id', '')):
            raise ValueError('Invalid schema or capture ID')
        metadata, identities = manifest['metadata'], manifest['identities']
        if identities.get('source') not in ('camera', 'replay') or metadata.get('source') != identities['source']:
            raise ValueError('Missing or inconsistent source provenance')
        started, finished, duration = (manifest[k] for k in ('started', 'finished_at', 'duration_s'))
        if not all(_number(x) for x in (started, finished, duration)) or not 0 < duration <= 60 or finished < started + duration:
            raise ValueError('Invalid capture timing')
        frames = manifest['frames']
        if not isinstance(frames, list) or not 1 <= len(frames) <= 140 or len(members) != len(frames) + 1:
            raise ValueError('Invalid frame count')
        if not isinstance(manifest['events'], list) or len(manifest['events']) > 512:
            raise ValueError('Invalid events')
        capture = CaptureBuffer(duration_s=duration)
        capture.id, capture.metadata, capture.events = manifest['id'], metadata, manifest['events']
        capture.started, capture.finished_at, capture.state = started, finished, 'exporting'
        total, previous = 0, None
        for index, (frame, member) in enumerate(zip(frames, members[1:])):
            shape = frame['shape']
            if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
                    or any(type(x) is not int or not 1 <= x <= 4096 for x in shape)):
                raise ValueError('Invalid RGB shape')
            size = math.prod(shape)
            total += size
            expected = f'frames/{index:04d}.rgb'
            if (frame['path'] != expected or member.name != expected or member.size != size
                    or frame['bytes'] != size or total > MAX_RGB):
                raise ValueError('Invalid frame path or size')
            row = frame['row']
            when, offset = row['frame_time'], row['offset_s']
            if (not _number(when) or not _number(offset) or not started <= when < started + duration
                    or (previous is not None and when <= previous) or abs((when - started) - offset) > 1e-6):
                raise ValueError('Invalid frame timing')
            previous = when
            raw = archive.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != frame['sha256']:
                raise ValueError('RGB hash mismatch')
            capture.frames.append((np.frombuffer(raw, np.uint8).reshape(shape).copy(), row))
        capture.count = len(frames)
        return capture, identities, manifest


def process_capture(path, output):
    capture, identities, manifest = load_capture(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / capture.id
    archive_digest = digest(path)
    if final.exists():
        old = json.loads((final / 'raw-provenance.json').read_text())
        if old.get('archive_sha256') != archive_digest:
            raise ValueError('Existing processed capture has a different archive identity')
        return json.loads((final / 'summary.json').read_text())
    staging = Path(tempfile.mkdtemp(prefix='.processing-', dir=output))
    summary = export_capture(capture, staging, identities)
    if not summary.get('export_complete'):
        raise ValueError('Review export did not complete')
    folder = staging / capture.id
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Recorded capture review</title>
<style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:16px;background:#edf3fa;color:#172c45}img{width:100%}pre{white-space:pre-wrap}button,select,input{font:inherit;margin:8px}</style>
<h1>Recorded capture review</h1><p>Masks are reconstructed on the laptop; telemetry is original recorded output.</p>
<select id="mode" aria-label="Image view"><option value="overlay">Selection boxes</option><option value="raw">Raw camera</option><option value="mask">Search masks</option><option value="tracker_mask">Tracker masks</option></select>
<button id="play">Play</button><input id="seek" aria-label="Frame" type="range" min="0" value="0"><span id="position"></span>
<img id="image" alt="Recorded frame"><pre id="telemetry"></pre><h2>Take results</h2><pre id="summary"></pre>
<script>const $=id=>document.getElementById(id);let rows=[],index=0,playing=false,timer;
function show(){if(!rows.length)return;const row=rows[index];$('image').src='./'+row[$('mode').value];$('seek').value=index;$('position').textContent=`Frame ${index+1}/${rows.length}, ${row.offset_s.toFixed(3)} s`;$('telemetry').textContent=JSON.stringify(row,null,2);}
function tick(){show();if(!playing)return;if(index+1>=rows.length){playing=false;$('play').textContent='Play';return;}const delay=(rows[index+1].offset_s-rows[index].offset_s)*1000;index++;timer=setTimeout(tick,Math.max(1,delay));}
$('play').onclick=()=>{playing=!playing;clearTimeout(timer);$('play').textContent=playing?'Pause':'Play';if(playing){if(index===rows.length-1)index=0;tick();}};
$('seek').oninput=()=>{playing=false;clearTimeout(timer);$('play').textContent='Play';index=+$('seek').value;show();};$('mode').onchange=show;
Promise.all([fetch('summary.json').then(r=>r.json()),fetch('frames.jsonl').then(r=>r.text())]).then(([s,t])=>{rows=t.trim().split('\\n').map(l=>JSON.parse(l));$('seek').max=rows.length-1;$('summary').textContent=JSON.stringify(s,null,2);show();}).catch(e=>{$('summary').textContent='Could not load capture: '+e.message;});</script></html>"""
    (folder / 'review.html').write_text(page)
    (folder / 'raw-manifest.json').write_text(json.dumps(manifest, allow_nan=False))
    (folder / 'raw-provenance.json').write_text(json.dumps(dict(
        archive_sha256=digest(path), archive_bytes=Path(path).stat().st_size,
        schema=SCHEMA, source=identities['source']), allow_nan=False))
    with zipfile.ZipFile(folder / 'download.zip', 'a', compression=zipfile.ZIP_STORED) as archive:
        for name in ('review.html', 'raw-manifest.json', 'raw-provenance.json'):
            archive.write(folder / name, name)
    folder.rename(final)
    staging.rmdir()
    return summary
