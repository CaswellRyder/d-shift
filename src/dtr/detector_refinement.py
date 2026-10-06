"""Bound a new goal refinement to a saved model and unchanged development labels."""

from pathlib import Path

from dtr.data import read_json, sha256


def refinement_source(snapshot, dataset_receipt, standard_path):
    snapshot = Path(snapshot).resolve()
    receipt_path = snapshot / "snapshot.json"
    receipt = read_json(receipt_path)
    if (receipt["kind"] != "interim_checkpoint_snapshot" or receipt["promotion_allowed"]
            or receipt["test_evaluated"] or receipt["deployment_approved"]):
        raise ValueError("Require an unpromoted development snapshot")
    model = snapshot / "model.pt"
    if sha256(model) != receipt["model_sha256"]:
        raise ValueError("Refinement checkpoint changed")
    parent = receipt["experiment"]
    if parent["standard_sha256"] != sha256(standard_path):
        raise ValueError("Refinement acceptance standard changed")
    parent_receipt_path = Path(receipt["dataset_root"]) / "receipt.json"
    if sha256(parent_receipt_path) != parent["dataset_receipt_sha256"]:
        raise ValueError("Parent dataset receipt changed")
    original = read_json(parent_receipt_path)
    for value in (original, dataset_receipt):
        if value["test_exported"] or set(value["splits"]) != {"train", "valid"}:
            raise ValueError("Refinement only permits train/valid exports")
        if value["classes"] != read_json(standard_path)["classes"]:
            raise ValueError("Refinement goal class order changed")
    if original["splits"]["valid"] != dataset_receipt["splits"]["valid"]:
        raise ValueError("Refinement must preserve validation frames and labels")
    return dict(snapshot=str(snapshot), snapshot_receipt_sha256=sha256(receipt_path),
                model_sha256=receipt["model_sha256"], parent_epoch=receipt["epoch"],
                parent_dataset_receipt_sha256=parent["dataset_receipt_sha256"],
                kind="new_optimizer_from_development_selected_ema",
                scope="Compound refinement, not a label-only causal comparison")


def training_options(refinement=None):
    if refinement is None:
        return dict(lr0=0.001, lrf=0.01, warmup_epochs=2, mosaic=0.5, close_mosaic=5)
    return dict(lr0=0.0001, lrf=0.01, warmup_epochs=0, mosaic=0.0, close_mosaic=0)
