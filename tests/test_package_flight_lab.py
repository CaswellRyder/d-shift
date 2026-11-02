"""Generated package fixtures only: no models, Pi, network or operator recordings."""

import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts import package_flight_lab as kit

REVISION = 'a' * 40


def put(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path / 'repo with spaces'
    repo.mkdir()
    for relative in kit.PI_FILES:
        put(repo, relative, 'source ' + relative)
    put(repo, 'src/dtr/flight.py', 'flight source')
    put(repo, 'src/dtr/__pycache__/private.pyc', b'not public')
    for relative in kit.TEMPLATES:
        put(repo, 'lab_kit/' + relative,
            '{"host":"vision.local","user":"pacman","port":8770}'
            if relative == 'connection.example.json' else 'template ' + relative)
    for relative in kit.DOCUMENTS:
        put(repo, 'lab_kit/' + relative, 'public setup documentation')
    put(repo, 'lab_kit/Laptop/tests/test_portable.py', 'test source')
    for relative in ('Evidence/Conversation archive.html', 'class-rulebook.docx',
                     'Source Snapshot/private.txt', 'Captures/recording.png',
                     'Examples/synthetic.png', 'Laptop/.runtime.json', '.env'):
        put(repo, 'lab_kit/' + relative, 'private content')
    put(repo, 'data/IMG_reserved.png', 'reserved data')
    runtime = put(tmp_path, 'input/runtime.so', b'frozen synthetic ARM runtime')
    put(repo, kit.RUNTIME_EVIDENCE,
        json.dumps({'files': {'build-armv6/libtensorflowlite_c.so': kit.digest(runtime)}}))
    models = {}
    for kind in ('balloon', 'goal'):
        model = put(tmp_path, f'input/{kind}.tflite', (kind + ' generated numerical model').encode())
        put(tmp_path, f'input/{kind}.json', json.dumps(dict(
            sha256=kit.digest(model), deployment_approved=False,
            classes=['background', kind], provenance='unchanged synthetic fixture')) + '\n')
        models[kind] = model
    return dict(repo=repo, balloon_model=models['balloon'], goal_model=models['goal'],
                runtime=runtime, output=tmp_path / 'transfer kit', source_revision=REVISION)


def build(fixture):
    return kit.package(**fixture)


def test_package_contains_only_allowed_material_and_preserves_models(fixture):
    result = build(fixture)
    root = Path(result['folder'])
    names = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert not any('Conversation' in name or 'rulebook' in name or 'Source Snapshot' in name
                   or '__pycache__' in name or name.endswith('.env') or 'Examples/' in name for name in names)
    assert not (root / 'Laptop/.runtime.json').exists()
    assert list((root / 'Captures').iterdir()) == [root / 'Captures/README.txt']
    assert (root / 'Pi Bundle/models/balloon.json').read_bytes() == fixture['balloon_model'].with_suffix('.json').read_bytes()
    assert (root / 'Pi Bundle/models/goal.json').read_bytes() == fixture['goal_model'].with_suffix('.json').read_bytes()
    assert (root / 'Pi Bundle/runtime/libtensorflowlite_c.so').read_bytes() == fixture['runtime'].read_bytes()
    release = json.loads((root / 'RELEASE-MANIFEST.json').read_text())
    assert release['source_revision'] == REVISION
    assert release['deployment_approved'] is False
    assert release['includes_recordings'] is release['includes_datasets'] is False
    assert release['includes_conversation'] is release['includes_class_rulebook'] is False
    assert 'input/' not in (root / 'RELEASE-MANIFEST.json').read_text()


def test_all_archive_files_have_exact_hashes_and_commands_are_executable(fixture, tmp_path):
    result = build(fixture)
    root = Path(result['folder'])
    bundle = root / 'Pi Bundle'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['archive_sha256'] == kit.digest(bundle / 'pi-bundle.tar.gz')
    with tarfile.open(bundle / 'pi-bundle.tar.gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert set(archive.getnames()) == set(manifest['files'])
        for member in members:
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == manifest['files'][member.name]
    sums = json.loads((root / 'SHA256SUMS.json').read_text())
    assert 'connection.json' not in sums['files']
    assert all(not name.startswith('Captures/') for name in sums['files'])
    with zipfile.ZipFile(result['zip']) as archive:
        assert archive.testzip() is None
        for relative, expected in sums['files'].items():
            assert hashlib.sha256(archive.read(root.name + '/' + relative)).hexdigest() == expected
        command = archive.getinfo(root.name + '/Start Flight Lab.command')
        assert command.external_attr >> 16 & 0o111
    assert (root / 'Start Flight Lab.command').stat().st_mode & 0o111
    relocated = tmp_path / 'other machine'
    root.rename(relocated)
    for relative, expected in sums['files'].items():
        assert kit.digest(relocated / relative) == expected


def test_pi_archive_is_deterministic_across_output_locations(fixture):
    first = build(fixture)
    second = build(dict(fixture, output=fixture['output'].with_name('another kit')))
    assert first['pi_archive_sha256'] == second['pi_archive_sha256']


@pytest.mark.parametrize('artifact', ['runtime', 'balloon_model', 'metadata'])
def test_changed_input_during_copy_cannot_publish_inconsistent_artifacts(fixture, monkeypatch, artifact):
    source = (fixture['balloon_model'].with_suffix('.json') if artifact == 'metadata'
              else fixture[artifact])
    original = kit.copy_file
    changed = False

    def replace_before_copy(path, destination, *args):
        nonlocal changed
        if path == source and not changed:
            source.write_bytes(b'changed after validation')
            changed = True
        return original(path, destination, *args)

    monkeypatch.setattr(kit, 'copy_file', replace_before_copy)
    with pytest.raises(ValueError, match='Staged'):
        build(fixture)
    assert changed
    assert not fixture['output'].exists()
    assert not fixture['output'].with_suffix('.zip').exists()


@pytest.mark.parametrize('kind', ['balloon_model', 'goal_model', 'runtime'])
def test_wrong_artifact_checksum_rejects_before_creating_output(fixture, kind):
    fixture[kind].write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum|runtime'):
        build(fixture)
    assert not fixture['output'].exists()
    assert not fixture['output'].with_suffix('.zip').exists()


@pytest.mark.parametrize('target', ['folder', 'zip'])
def test_existing_destination_is_never_overwritten(fixture, target):
    if target == 'folder':
        fixture['output'].mkdir()
        existing = put(fixture['output'], 'owned.txt', 'keep')
    else:
        existing = fixture['output'].with_suffix('.zip')
        existing.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        build(fixture)
    assert existing.read_bytes() == b'keep'


def test_missing_template_cleans_private_staging_without_publishing(fixture):
    (fixture['repo'] / 'lab_kit/Laptop/flight_lab.py').unlink()
    with pytest.raises(ValueError, match='Required regular file'):
        build(fixture)
    assert not fixture['output'].exists()
    assert not fixture['output'].with_suffix('.zip').exists()
    assert not list(fixture['output'].parent.glob('.transfer kit-build-*'))


def test_symlink_source_is_rejected(fixture):
    path = fixture['repo'] / 'src/dtr/flight.py'
    path.unlink()
    path.symlink_to(fixture['runtime'])
    with pytest.raises(ValueError, match='Required regular file'):
        build(fixture)


def test_parent_symlink_cannot_copy_files_from_outside_repo(fixture, tmp_path):
    directory = fixture['repo'] / 'lab_kit/Laptop'
    moved = tmp_path / 'outside source'
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes its repository'):
        build(fixture)
    assert not fixture['output'].exists()


def test_concurrent_empty_output_directory_is_not_replaced(fixture, monkeypatch):
    original = kit.tar_bundle

    def while_building(*args):
        fixture['output'].mkdir()
        return original(*args)

    monkeypatch.setattr(kit, 'tar_bundle', while_building)
    with pytest.raises(FileExistsError):
        build(fixture)
    assert list(fixture['output'].iterdir()) == []
    assert not fixture['output'].with_suffix('.zip').exists()


@pytest.mark.parametrize('revision', ['main', 'abc123', 'A' * 40, '../secret'])
def test_revision_must_identify_an_immutable_commit(fixture, revision):
    fixture['source_revision'] = revision
    with pytest.raises(ValueError, match='immutable'):
        build(fixture)
