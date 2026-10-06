"""Resource preflights for long local training jobs; never delete user files."""

from pathlib import Path
import shutil


class DiskSpaceError(RuntimeError):
    pass


class TrainingSegmentComplete(RuntimeError):
    """A saved epoch boundary, not completed training or a failed experiment."""


def require_segment_boundary(start_epoch, epoch, maximum_epochs, stopped):
    if maximum_epochs < 0:
        raise ValueError("Process epoch limit must be nonnegative")
    if maximum_epochs and not stopped and epoch - start_epoch + 1 >= maximum_epochs:
        raise TrainingSegmentComplete("Saved process epoch limit reached; resume in a fresh process")


def resume_stopper_state(checkpoint, saved, checkpoint_sha256):
    """Keep patience across process restarts; refuse ambiguous legacy state."""
    if saved is not None:
        if saved["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("Early-stopping state does not match saved checkpoint")
        return saved["early_stopping"]
    if checkpoint["train_metrics"]["fitness"] != checkpoint["best_fitness"]:
        raise ValueError("Legacy resume lacks exact early-stopping state; cannot infer best epoch")
    return dict(best_fitness=checkpoint["best_fitness"], best_epoch=checkpoint["epoch"] + 1,
                possible_stop=False)


def require_disk_space(paths, minimum_free_gib=10):
    if minimum_free_gib <= 0:
        raise ValueError("Free-space reserve must be positive")
    for path in paths:
        existing = Path(path).resolve()
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free / 1024**3
        if free < minimum_free_gib:
            raise DiskSpaceError(
                f"Training storage guard: {existing} has {free:.2f} GiB free; "
                f"requires at least {minimum_free_gib} GiB. Free space before resuming. "
                "No files were deleted."
            )
