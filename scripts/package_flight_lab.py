"""Build a public-safe, self-contained laptop/Pi kit. No network or inference."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PI_FILES = ('src/dtr/static/flight_lab.html', 'scripts/pi_flight_loop.py',
            'scripts/pi_flight_lab.py', 'scripts/calibrate_lens.py',
            'scripts/lab_session.py',
            'configs/camera_mount.json', 'configs/target_sizes.json',
            'LICENSE', 'THIRD_PARTY_NOTICES.txt')
TEMPLATES = ('Laptop/flight_lab.py', 'Laptop/pi_helper.py',
             'Start Flight Lab.command', 'Download Captures.command',
             'Verify Package.command', 'connection.example.json')
DOCUMENTS = ('START HERE.html',)
RUNTIME_EVIDENCE = 'models/v1.0.0/evidence/runtime-build.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def copy_file(source, destination, source_root=None):
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'Required regular file is missing: {source}')
    if source_root is not None:
        try:
            source.resolve().relative_to(source_root)
        except ValueError:
            raise ValueError('Source path escapes its repository: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o755 if destination.suffix == '.command' else 0o644)


def tar_bundle(bundle, paths):
    archive = bundle / 'pi-bundle.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as stream:
            for path in paths:
                info = stream.gettarinfo(str(path), arcname=path.relative_to(bundle).as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as contents:
                    stream.addfile(info, contents)
    return archive


def runtime_calibration(path):
    """Keep runtime geometry and source identity; omit local capture paths."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Lens calibration must be a regular file')
    raw = path.read_bytes()
    data = json.loads(raw)
    try:
        k, d, size, rms = (data[key] for key in ('K', 'D', 'image_size', 'rms'))
        numbers = [v for row in k for v in row] + list(d) + [rms]
        valid = (data.get('flipped') is True and len(k) == 3
                 and all(len(row) == 3 for row in k) and len(d) == 4
                 and size == [320, 240] and k[0][0] > 0 and k[1][1] > 0 and rms >= 0
                 and all(type(v) in (int, float) and math.isfinite(v) for v in numbers))
    except (TypeError, KeyError, IndexError):
        valid = False
    if not valid:
        raise ValueError('Expected finite calibration for flipped 320x240 runtime frames')
    return dict(K=k, D=d, image_size=size, rms=rms, flipped=True,
                status='provisional; independent physical validation still required',
                source_calibration_sha256=hashlib.sha256(raw).hexdigest())


def package(repo, balloon_model, goal_model, runtime, output, source_revision,
            lens_calibration=None):
    """Read only explicit source/templates and supplied artifacts; return transfer paths."""
    repo, output = Path(repo).resolve(), Path(output).absolute()
    archive_path = output.with_suffix('.zip')
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', source_revision):
        raise ValueError('source_revision must be a full immutable Git commit SHA')
    if output.exists() or output.is_symlink() or archive_path.exists() or archive_path.is_symlink():
        raise FileExistsError('Output folder or ZIP already exists; neither will be overwritten')
    lens = runtime_calibration(lens_calibration) if lens_calibration is not None else None
    models = {'balloon': Path(balloon_model), 'goal': Path(goal_model)}
    model_pins = {}
    for kind, path in models.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Model must be a regular file')
        metadata = path.with_suffix('.json')
        if metadata.is_symlink() or not metadata.is_file():
            raise ValueError('Model metadata must be a regular sibling JSON file')
        metadata_bytes = metadata.read_bytes()
        weight_hash = digest(path)
        if json.loads(metadata_bytes).get('sha256') != weight_hash:
            raise ValueError('Model checksum differs from its original metadata: ' + path.name)
        model_pins[kind] = (weight_hash, hashlib.sha256(metadata_bytes).hexdigest())
    runtime = Path(runtime)
    expected_runtime = json.loads((repo / RUNTIME_EVIDENCE).read_text())['files']['build-armv6/libtensorflowlite_c.so']
    if runtime.is_symlink() or not runtime.is_file() or digest(runtime) != expected_runtime:
        raise ValueError('Native runtime does not match the frozen ARMv6 runtime evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-build-', dir=output.parent))
    temporary_zip = staging.parent / (staging.name + '.zip')
    try:
        templates = repo / 'lab_kit'
        for relative in TEMPLATES:
            copy_file(templates / relative, staging / relative, repo)
        for relative in DOCUMENTS:
            copy_file(templates / relative, staging / relative, repo)
        copy_file(staging / 'connection.example.json', staging / 'connection.json')
        bundle = staging / 'Pi Bundle'
        for relative in PI_FILES:
            copy_file(repo / relative, bundle / relative, repo)
        copy_file(repo / 'scripts/lab_session.py', staging / 'Laptop/lab_session.py', repo)
        if lens is not None:
            if digest(lens_calibration) != lens['source_calibration_sha256']:
                raise ValueError('Lens calibration changed after validation')
            write_json(bundle / 'configs/lens-calibration.json', lens)
        for path in sorted((repo / 'src/dtr').glob('*.py')):
            copy_file(path, bundle / 'src/dtr' / path.name, repo)
        copy_file(repo / RUNTIME_EVIDENCE, bundle / 'runtime/runtime-build.json', repo)
        for kind, path in models.items():
            copy_file(path, bundle / 'models' / (kind + '.tflite'))
            copy_file(path.with_suffix('.json'), bundle / 'models' / (kind + '.json'))
        copy_file(runtime, bundle / 'runtime/libtensorflowlite_c.so')
        # Inputs may change while source files are copied. Bind the archive to the
        # validated input bytes, including metadata flags, before publishing it.
        for kind, (weight_hash, metadata_hash) in model_pins.items():
            if (digest(bundle / 'models' / (kind + '.tflite')) != weight_hash
                    or digest(bundle / 'models' / (kind + '.json')) != metadata_hash):
                raise ValueError('Staged model or metadata changed after validation: ' + kind)
        if digest(bundle / 'runtime/libtensorflowlite_c.so') != expected_runtime:
            raise ValueError('Staged native runtime differs from frozen evidence')
        paths = sorted(p for p in bundle.rglob('*') if p.is_file())
        identities = {p.relative_to(bundle).as_posix(): digest(p) for p in paths}
        pi_archive = tar_bundle(bundle, paths)
        manifest = dict(schema=1, files=identities, archive_sha256=digest(pi_archive),
                        model_files=dict(balloon='models/balloon.tflite', goal='models/goal.tflite'),
                        runtime_file='runtime/libtensorflowlite_c.so', deployment_approved=False,
                        hardware='Existing configured Raspberry Pi Zero W ARMv6')
        if lens is not None:
            manifest['lens_calibration_file'] = 'configs/lens-calibration.json'
        write_json(bundle / 'manifest.json', manifest)
        write_json(staging / 'RELEASE-MANIFEST.json', dict(
            schema=1, source_revision=source_revision, status='research lab prerelease',
            deployment_approved=False, includes_datasets=False, includes_recordings=False,
            includes_conversation=False, includes_class_rulebook=False,
            model_sha256={kind: digest(bundle / 'models' / (kind + '.tflite')) for kind in models},
            runtime_sha256=expected_runtime, pi_archive_sha256=manifest['archive_sha256']))
        (staging / 'Captures').mkdir()
        (staging / 'Captures/README.txt').write_text('Future laptop downloads go here. No recordings are included.\n')
        checksums = {p.relative_to(staging).as_posix(): digest(p) for p in sorted(staging.rglob('*'))
                     if p.is_file() and p.name != 'connection.json' and 'Captures' not in p.relative_to(staging).parts}
        write_json(staging / 'SHA256SUMS.json', dict(schema=1, files=checksums, excluded=[
            'SHA256SUMS.json', 'connection.json', 'Captures/**',
            'Laptop/.runtime.json', 'Laptop/.runtime.json.tmp']))
        with zipfile.ZipFile(temporary_zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob('*')):
                if path.is_file():
                    info = zipfile.ZipInfo((Path(output.name) / path.relative_to(staging)).as_posix())
                    info.create_system = 3
                    info.external_attr = (0o100755 if path.suffix == '.command' else 0o100644) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
        output.mkdir()  # Exclusive claim, including when a concurrent empty directory appears.
        for child in staging.iterdir():
            child.rename(output / child.name)
        staging.rmdir()
        os.link(temporary_zip, archive_path)  # Exclusive publication; never replace a concurrent ZIP.
        temporary_zip.unlink()
        return dict(folder=str(output), zip=str(archive_path), zip_sha256=digest(archive_path),
                    zip_bytes=archive_path.stat().st_size, source_revision=source_revision,
                    pi_archive_sha256=manifest['archive_sha256'], immutable_files=len(checksums))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        temporary_zip.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--balloon-model', type=Path, required=True)
    parser.add_argument('--goal-model', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--lens-calibration', type=Path,
                        help='Explicitly include and use this provisional lens in the new kit')
    args = parser.parse_args()
    try:
        print(json.dumps(package(ROOT, args.balloon_model, args.goal_model, args.runtime,
                                 args.output, args.source_revision, args.lens_calibration), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
