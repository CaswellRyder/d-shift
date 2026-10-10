import json

import pytest

from dtr.data import sha256, write_json
from scripts.build_development_panel_bundle import bundle


def make_queue(tmp_path):
    queue = tmp_path / "queue"
    (queue / "frames").mkdir(parents=True)
    rows = []
    for i in range(3):
        path = queue / f"frames/{i:03d}.png"
        path.write_bytes(bytes([i]) * 8)
        rows.append(dict(id=i, path=f"frames/{i:03d}.png", sha256=sha256(path), source=f"IMG_{i}.jpg",
                         provisional_truth=[dict(label="red_balloon", box=[0, 0, 4, 4])] if i else []))
    write_json(queue / "review.json", dict(frames=rows))
    return queue


def make_review(tmp_path, queue, **changes):
    review = dict(queue_sha256=sha256(queue / "review.json"), evaluation_approved=True,
                  training_approved=False, admit=[0, 2], exclude=[1])
    review.update(changes)
    write_json(tmp_path / "review.json", review)
    return tmp_path / "review.json"


def test_bundle_copies_only_admitted_frames_with_truth(tmp_path):
    queue = make_queue(tmp_path)
    frames = bundle(queue, make_review(tmp_path, queue), tmp_path / "out")
    assert [f["queue_id"] for f in frames] == [0, 2]
    assert sorted(p.name for p in (tmp_path / "out/frames").iterdir()) == ["000.png", "002.png"]
    inputs = json.loads((tmp_path / "out/inputs.json").read_text())
    assert inputs["frames"][0]["truth"] == [] and len(inputs["frames"][1]["truth"]) == 1


@pytest.mark.parametrize("changes", [dict(training_approved=True), dict(evaluation_approved=False),
                                     dict(queue_sha256="0" * 64), dict(admit=[0, 1], exclude=[1])])
def test_bundle_rejects_unsafe_reviews(tmp_path, changes):
    queue = make_queue(tmp_path)
    with pytest.raises(ValueError):
        bundle(queue, make_review(tmp_path, queue, **changes), tmp_path / "out")


def test_bundle_rejects_changed_frames(tmp_path):
    queue = make_queue(tmp_path)
    review = make_review(tmp_path, queue)
    (queue / "frames/002.png").write_bytes(b"changed")
    with pytest.raises(ValueError):
        bundle(queue, review, tmp_path / "out")
