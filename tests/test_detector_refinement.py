from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from dtr.data import read_json, sha256, write_json
from dtr.detector_refinement import refinement_source, training_options


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.standard = self.root / "standard.json"
        write_json(self.standard, dict(classes=["orange_circle"]))
        self.dataset = dict(classes=["orange_circle"], test_exported=False,
                            splits=dict(train={"counts": 1}, valid={"annotation_sha256": "fixed"}))
        write_json(self.root / "receipt.json", self.dataset)
        (self.root / "model.pt").write_bytes(b"fixture only, never deserialized")
        self.receipt = dict(kind="interim_checkpoint_snapshot", epoch=17,
                            promotion_allowed=False, test_evaluated=False, deployment_approved=False,
                            dataset_root=str(self.root), model_sha256=sha256(self.root / "model.pt"),
                            experiment=dict(standard_sha256=sha256(self.standard),
                                            dataset_receipt_sha256=sha256(self.root / "receipt.json")))
        write_json(self.root / "snapshot.json", self.receipt)

    def resolve(self):
        return refinement_source(self.root, self.dataset, self.standard)

    def test_only_training_labels_may_change(self):
        self.dataset["splits"]["train"] = {"counts": 2}
        result = self.resolve()
        self.assertEqual(result["parent_epoch"], 17)
        self.assertEqual(result["model_sha256"], self.receipt["model_sha256"])
        self.assertIn("new_optimizer", result["kind"])

    def test_validation_change_refused(self):
        self.dataset["splits"]["valid"]["annotation_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "preserve validation"):
            self.resolve()

    def test_model_and_receipt_tampering_refused(self):
        for filename in ("model.pt", "receipt.json", "standard.json"):
            with self.subTest(filename=filename):
                path = self.root / filename
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    self.resolve()
                path.write_bytes(original)

    def test_class_order_and_test_export_refused(self):
        for change in (dict(classes=["wrong"]), dict(test_exported=True)):
            original = deepcopy(self.dataset)
            self.dataset.update(change)
            with self.assertRaises(ValueError):
                self.resolve()
            self.dataset = original

    def test_promoted_or_test_evaluated_snapshot_refused(self):
        for key in ("promotion_allowed", "deployment_approved", "test_evaluated"):
            receipt = deepcopy(self.receipt)
            receipt[key] = True
            write_json(self.root / "snapshot.json", receipt)
            with self.assertRaises(ValueError):
                self.resolve()

    def test_profiles_are_distinct_and_baseline_unchanged(self):
        self.assertEqual(training_options(), dict(lr0=.001, lrf=.01, warmup_epochs=2,
                                                  mosaic=.5, close_mosaic=5))
        options = training_options(self.resolve())
        self.assertEqual(options["lr0"], .0001)
        self.assertEqual(options["warmup_epochs"], 0)
        self.assertEqual(options["mosaic"], 0)
        self.assertEqual(read_json(self.root / "receipt.json"), self.dataset)


if __name__ == "__main__":
    unittest.main()
