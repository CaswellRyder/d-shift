"""Run with unittest in .detector-venv; skip in the separate Keras environment."""

import importlib.util
import io
import random
from types import SimpleNamespace
import unittest

import numpy as np

from dtr.training_random import capture_random_state, restore_random_state, seed_migrated_segment


@unittest.skipUnless(importlib.util.find_spec("torch"), "Separate detector environment")
class TrainingRandomTests(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest("MPS unavailable")
        self.torch = torch
        self.loaders = {name: SimpleNamespace(generator=torch.Generator()) for name in ("train", "valid")}

    def draw(self):
        t = self.torch
        return [random.random(), float(np.random.random()), t.rand(4), t.rand(4, device="mps").cpu(),
                t.randperm(20, generator=self.loaders["train"].generator),
                t.randperm(20, generator=self.loaders["valid"].generator)]

    def test_restore_all_random_streams(self):
        seed_migrated_segment(self.torch, self.loaders, 42, 14)
        state = capture_random_state(self.torch, self.loaders)
        serialized = io.BytesIO()
        self.torch.save(state, serialized)
        serialized.seek(0)
        state = self.torch.load(serialized, map_location="cpu", weights_only=False)
        expected = self.draw()
        self.draw()
        restore_random_state(self.torch, self.loaders, state)
        actual = self.draw()
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(expected[2:], actual[2:])))

    def test_legacy_segments_do_not_repeat_shuffle(self):
        seed_migrated_segment(self.torch, self.loaders, 42, 14)
        a = self.draw()
        seed_migrated_segment(self.torch, self.loaders, 42, 15)
        b = self.draw()
        self.assertNotEqual(a[:2], b[:2])
        self.assertFalse(self.torch.equal(a[-2], b[-2]))

    def test_loader_mismatch_refuses_restore(self):
        state = capture_random_state(self.torch, self.loaders)
        with self.assertRaises(ValueError):
            restore_random_state(self.torch, {}, state)


if __name__ == "__main__":
    unittest.main()
