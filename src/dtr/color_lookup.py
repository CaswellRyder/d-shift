"""Optional exact 24-bit RGB goal-mask lookup: 16 MiB, never downloads/generates at runtime."""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from .data import read_json, sha256


class GoalColorLookup:
    def __init__(self, path):
        path = Path(path)
        meta = read_json(path.with_suffix(".json"))
        if (meta.get("contract") != "dtr-goal-rgb24-hsv-v1"
                or path.stat().st_size != 1 << 24 or sha256(path) != meta["sha256"]):
            raise ValueError("Goal lookup size/contract/hash mismatch")
        self.values = np.fromfile(path, dtype=np.uint8)
        if np.any(self.values > 3):
            raise ValueError("Invalid goal membership flags")
        self.values.flags.writeable = False

    def masks(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HWC uint8 RGB")
        # Local buffers keep shared lookup safe across concurrent callers.
        indices = rgb[:, :, 0].astype(np.uint32)
        np.left_shift(indices, 8, out=indices)
        np.bitwise_or(indices, rgb[:, :, 1], out=indices)
        np.left_shift(indices, 8, out=indices)
        np.bitwise_or(indices, rgb[:, :, 2], out=indices)
        flags = self.values[indices]
        return [(flags & 1)*np.uint8(255), (flags >> 1)*np.uint8(255)]


@lru_cache(maxsize=1)
def _lookup(path):
    return GoalColorLookup(path)


def goal_lookup_masks(rgb):
    path = os.environ.get("DTR_GOAL_COLOR_LOOKUP")
    if not path:
        raise ValueError("goal_lut requires explicit DTR_GOAL_COLOR_LOOKUP path")
    return _lookup(str(Path(path).resolve())).masks(rgb)
