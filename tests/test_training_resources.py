from collections import namedtuple

import pytest

from dtr.training_resources import (DiskSpaceError, TrainingSegmentComplete, require_disk_space,
                                    require_segment_boundary, resume_stopper_state)

Usage = namedtuple("Usage", "total used free")


def test_low_space_refuses_training_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr("dtr.training_resources.shutil.disk_usage", lambda p: Usage(20, 15, 5*1024**3))
    output = tmp_path / "new/run"
    with pytest.raises(DiskSpaceError, match="5.00 GiB"):
        require_disk_space([output])
    assert not output.exists()


def test_sufficient_output_space_does_not_ignore_full_temp_volume(tmp_path, monkeypatch):
    output, temp = tmp_path / "output", tmp_path / "temp"
    output.mkdir()
    temp.mkdir()
    monkeypatch.setattr("dtr.training_resources.shutil.disk_usage",
                        lambda p: Usage(20, 1, (15 if p == output else 2)*1024**3))
    with pytest.raises(DiskSpaceError, match="2.00 GiB"):
        require_disk_space([output, temp])


def test_reserve_boundary_and_invalid_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr("dtr.training_resources.shutil.disk_usage", lambda p: Usage(20, 10, 10*1024**3))
    require_disk_space([tmp_path])
    with pytest.raises(ValueError):
        require_disk_space([tmp_path], minimum_free_gib=0)


def test_process_epoch_budget_does_not_override_real_completion():
    require_segment_boundary(13, 13, 1, True)
    require_segment_boundary(13, 13, 0, False)
    require_segment_boundary(13, 13, 2, False)
    with pytest.raises(TrainingSegmentComplete):
        require_segment_boundary(13, 14, 2, False)
    with pytest.raises(ValueError):
        require_segment_boundary(13, 13, -1, False)


def test_resume_patience_uses_checksum_bound_state():
    saved = dict(checkpoint_sha256="abc", early_stopping=dict(
        best_epoch=11, best_fitness=.8, possible_stop=True))
    assert resume_stopper_state({}, saved, "abc") == saved["early_stopping"]
    with pytest.raises(ValueError, match="does not match"):
        resume_stopper_state({}, saved, "changed")


def test_legacy_patience_only_migrates_when_current_epoch_is_best():
    checkpoint = dict(epoch=12, best_fitness=.7, train_metrics=dict(fitness=.7))
    assert resume_stopper_state(checkpoint, None, "abc")["best_epoch"] == 13
    checkpoint["train_metrics"]["fitness"] = .6
    with pytest.raises(ValueError, match="cannot infer"):
        resume_stopper_state(checkpoint, None, "abc")
