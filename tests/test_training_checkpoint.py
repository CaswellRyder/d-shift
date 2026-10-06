"""Small-model continuation regression; does not establish full detector parity."""

from copy import deepcopy
import importlib.util
import io
from types import SimpleNamespace
import unittest

from dtr.training_checkpoint import capture_training_state, restore_training_state, same_state


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch environment required")
class TrainingCheckpointTests(unittest.TestCase):
    def trainer(self):
        import torch
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        return SimpleNamespace(model=model, optimizer=optimizer,
                               ema=SimpleNamespace(ema=deepcopy(model), updates=13),
                               scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9),
                               scaler=torch.amp.GradScaler("cpu", enabled=False))

    def step(self, trainer):
        import torch
        trainer.optimizer.zero_grad()
        trainer.model(torch.tensor([[.1, .2, .3]])).square().sum().backward()
        trainer.optimizer.step()
        trainer.scheduler.step()

    def test_serialized_resume_keeps_raw_weights_moments_and_next_update(self):
        import torch
        original = self.trainer()
        self.step(original)
        # EMA deliberately differs: restoring it as the training model would fail.
        state = capture_training_state(torch, original)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)
        resumed = self.trainer()
        ids = [id(p) for p in resumed.model.parameters()]
        restore_training_state(resumed, loaded)
        self.assertEqual(ids, [id(p) for p in resumed.model.parameters()])
        self.assertEqual(resumed.ema.updates, 13)
        self.assertEqual(resumed.scheduler.state_dict(), original.scheduler.state_dict())
        for a, b in zip(original.model.parameters(), resumed.model.parameters()):
            self.assertTrue(torch.equal(a, b))
        for item in resumed.optimizer.state.values():
            self.assertEqual(item["exp_avg"].dtype, torch.float32)
            self.assertEqual(item["exp_avg_sq"].dtype, torch.float32)
        self.step(original)
        self.step(resumed)
        for a, b in zip(original.model.parameters(), resumed.model.parameters()):
            self.assertTrue(torch.equal(a, b))
        # Captured CPU tensors must not alias the live model.
        self.assertFalse(torch.equal(state["model"]["weight"], original.model.weight))

    def test_unknown_format_refused(self):
        with self.assertRaises(ValueError):
            restore_training_state(self.trainer(), {"format": "unknown"})

    def test_exact_comparison_rejects_precision_and_missing_state(self):
        import torch
        state = capture_training_state(torch, self.trainer())
        self.assertTrue(same_state(torch, state, deepcopy(state)))
        changed = deepcopy(state)
        changed["model"]["weight"] = changed["model"]["weight"].half()
        self.assertFalse(same_state(torch, state, changed))
        changed = deepcopy(state)
        del changed["optimizer"]
        self.assertFalse(same_state(torch, state, changed))


if __name__ == "__main__":
    unittest.main()
