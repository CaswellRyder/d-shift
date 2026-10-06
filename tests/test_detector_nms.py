"""Run in .detector-venv with unittest; optional in the main Keras environment."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("ultralytics"), "Separate detector environment")
class DetectorNmsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["YOLO_CONFIG_DIR"] = str(Path("artifacts/detector-settings").resolve())
        os.environ["YOLO_OFFLINE"] = "true"
        import torch
        torch.set_num_threads(2)

    def batch(self):
        import torch
        # Two overlapping class-0 boxes plus a separate class-1 box, in every frame.
        return torch.tensor([[20., 21., 80.], [20., 21., 80.], [10., 10., 10.],
                             [10., 10., 10.], [.9, .8, .01], [.01, .01, .95]])[None].repeat(16, 1, 1)

    def test_complete_despite_elapsed_time_and_preserve_input(self):
        import torch
        from ultralytics.utils import nms
        from dtr.detector_nms import complete_cpu_nms
        raw = self.batch()
        original = raw.clone()
        kwargs = dict(conf_thres=.25, iou_thres=.5, multi_label=True)
        reference = nms.non_max_suppression(raw.clone(), **kwargs)
        # Reproduce the underlying cutoff deterministically: batches after the first are empty.
        with patch.object(nms.time, "time", side_effect=range(0, 100000, 100)):
            truncated = nms.non_max_suppression(raw.clone(), **kwargs)
        self.assertTrue(any(len(value) == 0 for value in truncated))
        with patch.object(nms.time, "time", side_effect=range(0, 100000, 100)):
            actual = complete_cpu_nms((raw, None), **kwargs)
        self.assertTrue(torch.equal(raw, original))
        self.assertEqual(len(actual), 16)
        for expected, value in zip(reference, actual):
            self.assertEqual(len(value), 2)
            self.assertTrue(torch.equal(expected, value))

    def test_mps_cpu_parity(self):
        import torch
        from dtr.detector_nms import complete_cpu_nms
        if not torch.backends.mps.is_available():
            self.skipTest("MPS unavailable")
        raw = self.batch()
        cpu = complete_cpu_nms(raw, conf_thres=.25, iou_thres=.5)
        mps = complete_cpu_nms(raw.to("mps"), conf_thres=.25, iou_thres=.5)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cpu, mps)))


if __name__ == "__main__":
    unittest.main()
