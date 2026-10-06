import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "snapshot_goal_detector", Path(__file__).parents[1] / "scripts/snapshot_goal_detector.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SnapshotEpochTests(unittest.TestCase):
    def test_resumable_epoch(self):
        self.assertEqual(module.saved_epoch({"epoch": 19}, False), 20)
        with self.assertRaises(ValueError):
            module.saved_epoch({"epoch": -1}, False)

    def test_finalized_last_epoch_requires_consistent_evidence(self):
        checkpoint = dict(epoch=-1, optimizer=None, ema=None, model="fixture",
                          train_results={"epoch": [1, 2, 3]})
        self.assertEqual(module.saved_epoch(checkpoint, True, 3), 3)
        for change, epoch in (({}, 4), ({"epoch": 2}, 3), ({"model": None}, 3),
                              ({"optimizer": {}}, 3), ({"train_results": {}}, 3)):
            with self.subTest(change=change, epoch=epoch), self.assertRaises(ValueError):
                module.saved_epoch({**checkpoint, **change}, True, epoch)


if __name__ == "__main__":
    unittest.main()
