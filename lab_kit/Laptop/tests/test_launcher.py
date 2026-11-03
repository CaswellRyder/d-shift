"""Dependency-free portability/setup checks. No network, Pi or camera access."""

import base64
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

LAPTOP = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load('portable_flight_lab', LAPTOP / 'flight_lab.py')
pi = load('portable_pi_helper', LAPTOP / 'pi_helper.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_kit(root):
    (root / 'Laptop').mkdir()
    shutil.copy2(LAPTOP / 'flight_lab.py', root / 'Laptop/flight_lab.py')
    shutil.copy2(LAPTOP / 'pi_helper.py', root / 'Laptop/pi_helper.py')
    (root / 'Pi Bundle').mkdir()
    files = {
        'scripts/pi_flight_lab.py': b'print("fixture")\n',
        'src/dtr/flight.py': b'# source fixture\n',
        'configs/camera_mount.json': b'{"calibrated":false}\n',
        'configs/target_sizes.json': b'{}\n',
        'models/balloon.tflite': b'balloon fixture',
        'models/balloon.json': b'{"fixture":"balloon"}\n',
        'models/goal.tflite': b'goal fixture',
        'models/goal.json': b'{"fixture":"goal"}\n',
        'runtime/libtensorflowlite_c.so': b'ARM fixture',
    }
    archive = root / 'Pi Bundle/pi-bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    manifest = dict(schema=1, files={k: sha(v) for k, v in files.items()},
                    archive_sha256=launcher.digest(archive),
                    model_files=dict(balloon='models/balloon.tflite', goal='models/goal.tflite'),
                    runtime_file='runtime/libtensorflowlite_c.so')
    (root / 'Pi Bundle/manifest.json').write_text(json.dumps(manifest))
    config = dict(host='vision.local', user='pacman', port=8770,
                  remote_root='/home/pacman/flight-lab-portable-20261102',
                  legacy_root='/home/pacman/flight-lab-20261102-44bfso60')
    (root / 'connection.json').write_text(json.dumps(config))
    sums = {str(p.relative_to(root)): launcher.digest(p)
            for p in root.rglob('*') if p.is_file()}
    (root / 'SHA256SUMS.json').write_text(json.dumps(dict(schema=1, files=sums)))
    return manifest, config, files


class KitCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='lab kit test with spaces ')
        self.root = Path(self.temp.name)
        self.manifest, self.config, self.files = make_kit(self.root)

    def tearDown(self):
        self.temp.cleanup()


class PackageTests(KitCase):
    def test_relocated_launcher_verifies_without_original_checkout(self):
        moved = self.root / 'moved folder with spaces'
        moved.mkdir()
        for name in ('Laptop', 'Pi Bundle'):
            shutil.copytree(self.root / name, moved / name)
        for name in ('connection.json', 'SHA256SUMS.json'):
            shutil.copy2(self.root / name, moved / name)
        result = subprocess.run([sys.executable, str(moved / 'Laptop/flight_lab.py'), '--check-package'],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Package verified', result.stdout)
        self.assertEqual(load('relocated_launcher', moved / 'Laptop/flight_lab.py').ROOT, moved.resolve())

    def test_tampered_source_is_rejected(self):
        (self.root / 'Laptop/pi_helper.py').write_text('changed')
        with self.assertRaisesRegex(launcher.LabError, 'integrity check failed'):
            launcher.check_package(self.root)

    def test_archive_digest_is_independently_checked(self):
        archive = self.root / 'Pi Bundle/pi-bundle.tar.gz'
        archive.write_bytes(b'changed')
        sums = json.loads((self.root / 'SHA256SUMS.json').read_text())
        sums['files']['Pi Bundle/pi-bundle.tar.gz'] = launcher.digest(archive)
        (self.root / 'SHA256SUMS.json').write_text(json.dumps(sums))
        with self.assertRaisesRegex(launcher.LabError, 'archive failed'):
            launcher.check_package(self.root)

    def test_help_never_connects_or_reads_package(self):
        with mock.patch.object(launcher, 'check_package', side_effect=AssertionError('package read')):
            with mock.patch.object(launcher.Connection, 'connect', side_effect=AssertionError('network')):
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit_:
                    launcher.main(['--help'])
        self.assertEqual(exit_.exception.code, 0)

    def test_verify_never_connects(self):
        with mock.patch.object(launcher, 'check_package', return_value=(self.manifest, 5)):
            with mock.patch.object(launcher.Connection, 'connect', side_effect=AssertionError('network')):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(['--check-package']), 0)

    def test_config_rejects_shell_and_ssh_option_injection(self):
        for key, value in [('host', '-oProxyCommand=bad'), ('host', 'vision.local; touch marker'),
                           ('user', 'pacman@other'), ('remote_root', '/home/pacman/../../etc'),
                           ('remote_root', '/home/pacman/a;bad'), ('port', True),
                           ('balloon_limit', True), ('balloon_limit', 0),
                           ('balloon_limit', 65), ('balloon_limit', '4')]:
            config = dict(self.config, **{key: value})
            (self.root / 'connection.json').write_text(json.dumps(config))
            with self.subTest(key=key, value=value), self.assertRaises(launcher.LabError):
                launcher.load_config(self.root)

    def test_payload_is_encoded_and_survives_quotes(self):
        payload = {'notes': "literal `command` $(command) ' \" ;\n"}
        command = launcher.remote_command(payload)
        self.assertTrue(command.startswith('python3 - '))
        encoded = command.split(' ', 2)[2]
        self.assertEqual(json.loads(base64.b64decode(encoded)), payload)
        self.assertNotIn('`command`', command)

    def test_no_remote_actions_on_local_integrity_failure(self):
        with mock.patch.object(launcher, 'check_package', side_effect=launcher.LabError('bad package')):
            with mock.patch.object(launcher.Connection, 'connect', side_effect=AssertionError('network')):
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(launcher.main([]), 1)

    def test_new_connection_keeps_interactive_host_key_checking(self):
        connection = launcher.Connection(self.root, self.config)
        test_socket = mock.MagicMock()
        test_socket.__enter__.return_value.connect_ex.return_value = 1
        with mock.patch.object(launcher.shutil, 'which', return_value='/usr/bin/ssh'):
            with mock.patch.object(launcher.socket, 'socket', return_value=test_socket):
                with mock.patch.object(launcher.subprocess, 'run', return_value=mock.Mock(returncode=0)) as run:
                    with redirect_stdout(io.StringIO()):
                        connection.connect()
        command = run.call_args.args[0]
        self.assertIn('StrictHostKeyChecking=ask', command)
        self.assertIn('-fN', command)
        self.assertNotIn('StrictHostKeyChecking=no', command)
        runtime = json.loads(connection.state_path.read_text())
        self.assertNotIn('password', runtime)
        shutil.rmtree(Path(connection.control).parent)

    def test_occupied_local_port_is_not_replaced(self):
        connection = launcher.Connection(self.root, self.config)
        test_socket = mock.MagicMock()
        test_socket.__enter__.return_value.connect_ex.return_value = 0
        with mock.patch.object(launcher.shutil, 'which', return_value='/usr/bin/ssh'):
            with mock.patch.object(launcher.socket, 'socket', return_value=test_socket):
                with mock.patch.object(launcher.subprocess, 'run', side_effect=AssertionError('ssh start')):
                    with self.assertRaisesRegex(launcher.LabError, 'already in use'):
                        connection.connect()


class DownloadTests(KitCase):
    def setUp(self):
        super().setUp()
        self.connection = launcher.Connection(self.root, self.config)
        self.connection.selected_root = self.config['legacy_root']
        self.ident = '20261102T190000-deadbeef'
        self.state = dict(capture=dict(state='idle'), sessions=[dict(id=self.ident)])

    def response(self, body, advertised_size=None):
        response = io.BytesIO(body)
        response.headers = {'Content-Length': str(len(body) if advertised_size is None else advertised_size)}
        return response

    def test_timed_capture_blocks_download_before_any_file_transfer(self):
        self.state['capture']['state'] = 'capturing'
        with mock.patch.object(launcher, 'urlopen', side_effect=AssertionError('transfer')):
            with self.assertRaisesRegex(launcher.LabError, 'Wait until'):
                self.connection.download(self.state)
        self.assertFalse((self.root / 'Captures').exists())

    def test_completed_take_uses_gated_http_zip_and_keeps_scope_index(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('summary.json', '{"complete":true}')
        body = data.getvalue()
        with mock.patch.object(launcher, 'urlopen', return_value=self.response(body)) as get:
            with redirect_stdout(io.StringIO()):
                output = self.connection.download(self.state)
        self.assertEqual(get.call_args.args[0], 'http://127.0.0.1:8770/sessions/' + self.ident + '/download.zip')
        self.assertEqual((output / (self.ident + '.zip')).read_bytes(), body)
        index = json.loads((output / 'download-index.json').read_text())
        self.assertTrue(index['complete'])
        self.assertIn('latest 100', index['scope'])
        self.assertEqual(index['files'][0]['sha256'], sha(body))

    def test_interrupted_download_remains_partial_and_index_is_not_complete(self):
        with mock.patch.object(launcher, 'urlopen', return_value=self.response(b'partial', 100)):
            with self.assertRaisesRegex(launcher.LabError, 'interrupted'):
                self.connection.download(self.state)
        output = next((self.root / 'Captures').iterdir())
        self.assertFalse((output / (self.ident + '.zip')).exists())
        self.assertTrue((output / (self.ident + '.zip.partial')).exists())
        self.assertFalse(json.loads((output / 'download-index.json').read_text())['complete'])

    def test_server_cannot_choose_a_download_path_outside_capture_folder(self):
        self.state['sessions'][0]['id'] = '../unrelated'
        with mock.patch.object(launcher, 'urlopen', side_effect=AssertionError('transfer')):
            with self.assertRaisesRegex(launcher.LabError, 'invalid capture identifier'):
                self.connection.download(self.state)


class PiHelperTests(KitCase):
    def install_payload(self, root):
        archive = self.root / 'upload.tar.gz'
        shutil.copy2(self.root / 'Pi Bundle/pi-bundle.tar.gz', archive)
        config = dict(self.config, remote_root=str(root))
        return dict(config=config, manifest=self.manifest, archive_path=str(archive))

    def test_fresh_install_is_self_contained_and_does_not_need_repo(self):
        root = self.root / 'pi install'
        result = pi.install(self.install_payload(root))
        self.assertEqual(result['root'], str(root))
        for name, data in self.files.items():
            self.assertEqual((root / name).read_bytes(), data)
        self.assertTrue((root / 'handoff-manifest.json').is_file())

    def test_existing_modified_install_is_not_overwritten(self):
        root = self.root / 'pi install'
        pi.install(self.install_payload(root))
        protected = root / 'src/dtr/flight.py'
        protected.write_text('owned modification')
        with self.assertRaisesRegex(pi.SetupError, 'differs'):
            pi.install(self.install_payload(root))
        self.assertEqual(protected.read_text(), 'owned modification')

    def test_symlink_archive_is_rejected_without_installing(self):
        payload = self.install_payload(self.root / 'blocked install')
        archive = Path(payload['archive_path'])
        with tarfile.open(archive, 'w:gz') as stream:
            info = tarfile.TarInfo('src/dtr/flight.py')
            info.type, info.linkname = tarfile.SYMTYPE, '/tmp/unrelated'
            stream.addfile(info)
        payload['manifest'] = dict(self.manifest, archive_sha256=pi.digest(archive))
        with self.assertRaisesRegex(pi.SetupError, 'link or special'):
            pi.install(payload)
        self.assertFalse(Path(payload['config']['remote_root']).exists())

    def test_unsafe_archive_member_is_rejected(self):
        payload = self.install_payload(self.root / 'blocked install')
        archive = Path(payload['archive_path'])
        with tarfile.open(archive, 'w:gz') as stream:
            info = tarfile.TarInfo('../escape')
            info.size = 1
            stream.addfile(info, io.BytesIO(b'x'))
        payload['manifest'] = dict(self.manifest, archive_sha256=pi.digest(archive))
        with self.assertRaisesRegex(pi.SetupError, 'Unsafe path'):
            pi.install(payload)
        self.assertFalse((self.root / 'escape').exists())

    def fake_running_process(self):
        root = self.root / 'legacy pi bundle'
        root.mkdir()
        for name, data in self.files.items():
            if name.startswith(('models/', 'runtime/')):
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        external = self.root / 'existing models and runtime'
        external.mkdir()
        for name, data in self.files.items():
            if name.startswith(('models/', 'runtime/')):
                path = external / Path(name).name
                path.write_bytes(data)
        argv = ['python3', '-u', str(root / 'scripts/pi_flight_lab.py'),
                '--balloon-model', str(external / 'balloon.tflite'),
                '--goal-model', str(external / 'goal.tflite')]
        (root / 'server.pid').write_text('4321')
        proc = self.root / 'proc/4321'
        proc.mkdir(parents=True)
        (proc / 'stat').write_text('4321 (python3) S 1 2 3')
        (proc / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in argv) + b'\0')
        (proc / 'environ').write_bytes(b'UNRELATED_SECRET=fixture-do-not-print\0DTR_TFLITE_LIBRARY=' +
                                      str(external / 'libtensorflowlite_c.so').encode() + b'\0')
        return root, external

    def test_running_legacy_bundle_models_and_runtime_are_verified(self):
        root, external = self.fake_running_process()
        with mock.patch.object(pi, 'PROC_ROOT', self.root / 'proc'):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(pi.validate_process(root, self.manifest), 4321)
            self.assertEqual(output.getvalue(), '')
            (external / 'libtensorflowlite_c.so').write_bytes(b'different runtime')
            with self.assertRaisesRegex(pi.SetupError, 'different native runtime'):
                pi.validate_process(root, self.manifest)

    def test_starting_viewer_on_a_different_port_is_not_reused(self):
        root, _ = self.fake_running_process()
        command = self.root / 'proc/4321/cmdline'
        command.write_bytes(command.read_bytes() + b'--port\0' + b'8770\0')
        config = dict(self.config, remote_root=str(self.root / 'new pi bundle'),
                      legacy_root=str(root), port=8771)
        with mock.patch.object(pi, 'PROC_ROOT', self.root / 'proc'), \
                mock.patch.object(pi, 'read_viewer_state', return_value=None), \
                mock.patch.object(pi.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(pi.SetupError, 'port 8770.*requests 8771'):
                pi.probe(dict(config=config, manifest=self.manifest))

    def test_reused_viewer_must_match_the_requested_balloon_limit(self):
        root, _ = self.fake_running_process()
        command = self.root / 'proc/4321/cmdline'
        base = command.read_bytes()
        with mock.patch.object(pi, 'PROC_ROOT', self.root / 'proc'):
            # Absent argument means the current script's default budget.
            self.assertEqual(pi.validate_process(root, self.manifest, expected_balloon_limit=4), 4321)
            with self.assertRaisesRegex(pi.SetupError, 'balloon_limit 4.*requests 2'):
                pi.validate_process(root, self.manifest, expected_balloon_limit=2)
            command.write_bytes(base + b'--balloon-limit\x008\x00')
            with self.assertRaisesRegex(pi.SetupError, 'balloon_limit 8.*requests 4'):
                pi.validate_process(root, self.manifest, expected_balloon_limit=4)
            self.assertEqual(pi.validate_process(root, self.manifest, expected_balloon_limit=8), 4321)
            command.write_bytes(base + b'--balloon-limit\x00many\x00')
            with self.assertRaisesRegex(pi.SetupError, 'invalid balloon limit'):
                pi.validate_process(root, self.manifest, expected_balloon_limit=4)

    def test_probe_refuses_to_reuse_a_viewer_with_another_budget(self):
        root, _ = self.fake_running_process()
        command = self.root / 'proc/4321/cmdline'
        command.write_bytes(command.read_bytes() + b'--balloon-limit\x008\x00')
        config = dict(self.config, remote_root=str(self.root / 'new pi bundle'),
                      legacy_root=str(root), balloon_limit=4)
        state = dict(source='camera', bundle=str(root), capture=dict(state='idle'))
        with mock.patch.object(pi, 'PROC_ROOT', self.root / 'proc'), \
                mock.patch.object(pi, 'read_viewer_state', return_value=state), \
                mock.patch.object(pi.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(pi.SetupError, 'balloon_limit 8.*requests 4'):
                pi.probe(dict(config=config, manifest=self.manifest))

    def test_unrelated_pid_is_never_reused(self):
        root, _ = self.fake_running_process()
        (self.root / 'proc/4321/cmdline').write_bytes(b'python3\0unrelated.py\0')
        with mock.patch.object(pi, 'PROC_ROOT', self.root / 'proc'):
            self.assertIsNone(pi.validate_process(root, self.manifest))

    def test_launch_uses_only_internal_bundle_models_and_runtime(self):
        root = self.root / 'self contained pi'
        pi.install(self.install_payload(root))
        payload = dict(config=dict(self.config, remote_root=str(root)), manifest=self.manifest,
                       wall_clock_offset_s=12.5)
        process = mock.Mock(pid=1234)
        with mock.patch.object(pi, 'probe', return_value=dict(kind='absent')):
            with mock.patch.object(pi.subprocess, 'Popen', return_value=process) as spawn:
                result = pi.start(payload)
        argv = spawn.call_args.args[0]
        env = spawn.call_args.kwargs['env']
        self.assertEqual(argv[argv.index('--balloon-model') + 1], str(root / 'models/balloon.tflite'))
        self.assertEqual(argv[argv.index('--goal-model') + 1], str(root / 'models/goal.tflite'))
        self.assertEqual(argv[argv.index('--balloon-limit') + 1], '4')
        self.assertEqual(env['DTR_TFLITE_LIBRARY'], str(root / 'runtime/libtensorflowlite_c.so'))
        self.assertFalse(any('/Users/' in arg for arg in argv))
        self.assertEqual(result['pid'], 1234)

    def test_existing_capture_is_reused_without_start_or_install(self):
        payload = dict(config=self.config, manifest=self.manifest)
        existing = dict(kind='existing', root=self.config['legacy_root'], capture_state='capturing')
        with mock.patch.object(pi, 'probe', return_value=existing):
            with mock.patch.object(pi.subprocess, 'Popen', side_effect=AssertionError('spawn')):
                self.assertEqual(pi.start(payload), existing)

    def test_capture_blocks_probe_before_source_hashes_read_from_disk(self):
        state = dict(source='camera', bundle=self.config['legacy_root'], capture=dict(state='capturing'))
        with mock.patch.object(pi, 'read_viewer_state', return_value=state):
            with mock.patch.object(pi, 'validate_process', side_effect=AssertionError('disk reads')):
                with self.assertRaisesRegex(pi.SetupError, 'current take'):
                    pi.probe(dict(config=self.config, manifest=self.manifest))

    def test_unrelated_remote_viewer_is_left_in_place(self):
        state = dict(source='camera', bundle='/home/pacman/other-project', capture=dict(state='idle'))
        with mock.patch.object(pi, 'read_viewer_state', return_value=state):
            with self.assertRaisesRegex(pi.SetupError, 'Another Pi bundle'):
                pi.probe(dict(config=self.config, manifest=self.manifest))


if __name__ == '__main__':
    unittest.main()
