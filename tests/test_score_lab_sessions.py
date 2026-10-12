import json

import pytest

from scripts import score_lab_sessions as lab


def det(label, accepted=True):
    return dict(label=label, accepted=accepted)


def write_scene(folder, rows, red=1, blue=0, training_use=False):
    folder.mkdir(parents=True)
    (folder / "scene.json").write_text(json.dumps(dict(
        scene=folder.name[:-3], take=1, notes="", model="m", model_sha256="x",
        red_in_view=red, blue_in_view=blue, training_use=training_use)))
    (folder / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_agreement_counts_extra_and_missing_per_label():
    frames = [[det("red_balloon")],
              [det("red_balloon"), det("red_balloon"), det("blue_balloon")],
              [det("red_balloon", accepted=False)],
              [det("background")]]
    result = lab.agreement(frames, {"red_balloon": 1, "blue_balloon": 0})
    assert result["exact_frames"] == 1 and result["exact_rate"] == 0.25
    assert result["red_balloon"] == dict(in_view=1, extra_frames=1, missing_frames=2, extra=1, missing=2)
    assert result["blue_balloon"]["extra"] == 1 and result["blue_balloon"]["missing"] == 0


def test_score_reads_scenes_and_timing(tmp_path):
    rows = [dict(detections=[det("red_balloon")], mode=mode, policy="periodic", processing_ms=10.0,
                 sensor_to_result_ms=20.0 + i, timestamp_valid=True, sensor_timestamp_ns=i * 200_000_000)
            for i, mode in enumerate(["mser_direct", "mser_bright", "mser_bright", "mser_direct"])]
    write_scene(tmp_path / "day" / "red-2m-01", rows)
    folders = lab.find_scenes([tmp_path])
    assert [f.name for f in folders] == ["red-2m-01"]
    (scene,) = lab.score(folders)
    assert scene["pi"]["exact_rate"] == 1.0
    assert scene["timing"]["fps"] == pytest.approx(5.0)
    assert scene["timing"]["full_search_fraction"] == 0.5
    assert scene["timing"]["policies"] == ["periodic"]
    assert "red-2m-01" in lab.table([scene])


def test_score_refuses_scenes_not_marked_evaluation_only(tmp_path):
    write_scene(tmp_path / "x-01", [], training_use=None)
    with pytest.raises(ValueError):
        lab.score([tmp_path / "x-01"])


def test_replay_checks_recovery_pixel_frames_against_pixel_search(tmp_path, monkeypatch):
    from PIL import Image

    def found(label):
        return [dict(box=[0, 0, 4, 4], crop_box=[0, 0, 4, 4], label=label, accepted=True,
                     raw_accepted=True, suppressed=False, scores=[0.0, 1.0, 0.0])]

    rows = [dict(path=f"{i:04d}.png", mode=mode, detections=found("red_balloon"))
            for i, mode in enumerate(["mser_direct", "pixel2", "pixel2"])]
    rows[2]["detections"] = found("blue_balloon")
    for row in rows:
        Image.new("RGB", (8, 8)).save(tmp_path / row["path"])
    modes = []

    def full(rgb, predictor, mode, native, selection):
        modes.append(mode)
        return dict(detections=found("red_balloon"))

    monkeypatch.setattr(lab, "process_frame", full)
    monkeypatch.setattr(lab, "pixel_frame", lambda *a, **k: dict(detections=found("red_balloon")))
    replay = object.__new__(lab.Replay)
    replay.masks = replay.native = None
    replay.predictor = lambda alias, digest: None
    replayed, parity = replay.scene(tmp_path, dict(model="m", model_sha256="x"), rows, ("parts", "pixel2"))
    assert modes == ["mser_direct"] * 3
    assert parity["frames"] == 3 and parity["mismatched_frames"] == 1
    assert len(replayed["pixel2"]) == 3
