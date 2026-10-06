import importlib.util
from pathlib import Path

import pytest

from dtr.data import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


MINE = module("mine_student_negatives")
REFINE = module("refine_pi_student")


def test_negative_overlap_guard_includes_context_and_border():
    candidate = {"crop_box": [10,10,20,20]}
    assert MINE.safe_background(candidate, [])
    assert MINE.safe_background(candidate, [[25,25,30,30]])
    for box in [[12,12,15,15], [5,5,30,30], [21,10,24,20]]:
        assert not MINE.safe_background(candidate, [box])


def test_train_source_selection_refuses_cross_split_and_traversal():
    row = dict(source_image="train/a.jpg", split="train", session="s")
    assert MINE.training_sources({"samples":[row]}) == {"train/a.jpg":"s"}
    for rows in [[row,dict(row,split="val")], [dict(row,source_image="valid/a.jpg")],
                 [dict(row,source_image="train/../test/a.jpg")]]:
        with pytest.raises(ValueError, match="boundary"):
            MINE.training_sources({"samples":rows})


def test_review_is_bound_to_exact_queue_manifest_and_image(tmp_path):
    crop = tmp_path/"crop.png"
    crop.write_bytes(b"fixture")
    queue = [dict(path="crop.png",sha256=sha256(crop),split="train",
                  source_image="train/a.jpg", label="background")]
    write_json(tmp_path/"queue.json",queue)
    digest = sha256(tmp_path/"queue.json")
    write_json(tmp_path/"receipt.json",dict(base_manifest_sha256="manifest",queue_sha256=digest))
    decision = tmp_path/"review.json"
    write_json(decision,dict(queue_sha256=digest,admit=[0]))
    assert REFINE.reviewed_negatives(tmp_path,decision,"manifest") == queue
    with pytest.raises(ValueError,match="hash mismatch"):
        REFINE.reviewed_negatives(tmp_path,decision,"changed")
    for indices in [[0,0],[-1],[1],[True],[]]:
        write_json(decision,dict(queue_sha256=digest,admit=indices))
        with pytest.raises(ValueError,match="indices"):
            REFINE.reviewed_negatives(tmp_path,decision,"manifest")
    write_json(decision,dict(queue_sha256=digest,admit=[0]))
    crop.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Invalid reviewed"):
        REFINE.reviewed_negatives(tmp_path,decision,"manifest")
