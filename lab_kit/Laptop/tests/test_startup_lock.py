"""A double click must never launch two camera processes. No SSH or Pi access."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('locked_pi_helper', Path(__file__).resolve().parents[1] / 'pi_helper.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class StartupLockTest(unittest.TestCase):
    def test_concurrent_start_refuses_second_camera_process(self):
        entered, release = threading.Event(), threading.Event()

        def launch(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError('Concurrent test did not release startup')
            return SimpleNamespace(pid=7000)

        with tempfile.TemporaryDirectory(prefix='portable startup test ') as directory:
            root = Path(directory)
            payload = dict(config=dict(remote_root=str(root), port=8770),
                           manifest=dict(runtime_file='runtime/libtensorflowlite_c.so',
                                         model_files=dict(balloon='models/balloon.tflite', goal='models/goal.tflite')),
                           wall_clock_offset_s=0.)
            with patch.object(helper, 'probe', return_value=dict(kind='absent')), \
                    patch.object(helper, 'verify_files'), \
                    patch.object(helper.subprocess, 'Popen', side_effect=launch) as popen, \
                    ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(helper.start, payload)
                try:
                    self.assertTrue(entered.wait(5))
                    with self.assertRaisesRegex(helper.SetupError, 'already in progress'):
                        helper.start(payload)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5)['pid'], 7000)
                self.assertEqual(popen.call_count, 1)
                self.assertEqual((root / 'server.pid').read_text(), '7000')


if __name__ == '__main__':
    unittest.main()
