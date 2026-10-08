"""Checked ctypes interface for an optional exact-mask library built on the target."""
import ctypes as ct
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import read_json, sha256


class NativeColorLookup:
    def __init__(self, path):
        path = Path(path)
        meta = read_json(path.with_suffix(".json"))
        if meta.get("contract") != "dtr-native-rgb24-v1" or sha256(path) != meta["sha256"]:
            raise ValueError("Native lookup contract/hash mismatch")
        self.lib = ct.CDLL(str(path.resolve()))
        self.lib.dtr_color_lookup_abi.argtypes = []
        self.lib.dtr_color_lookup_abi.restype = ct.c_uint32
        if self.lib.dtr_color_lookup_abi() != 1:
            raise ValueError("Native lookup ABI mismatch")
        byte_pointer = ct.POINTER(ct.c_uint8)
        self.lib.dtr_color_lookup.argtypes = [byte_pointer, ct.c_size_t, byte_pointer, byte_pointer, byte_pointer]
        self.lib.dtr_color_lookup.restype = None

    def masks(self, rgb, values):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HWC uint8 RGB")
        if values.dtype != np.uint8 or values.shape != (1 << 24,) or not values.flags.c_contiguous:
            raise ValueError("Expected contiguous RGB24 membership table")
        rgb = np.ascontiguousarray(rgb)
        red, blue = np.empty(rgb.shape[:2], np.uint8), np.empty(rgb.shape[:2], np.uint8)
        pointer = ct.POINTER(ct.c_uint8)
        self.lib.dtr_color_lookup(rgb.ctypes.data_as(pointer), red.size, values.ctypes.data_as(pointer),
                                  red.ctypes.data_as(pointer), blue.ctypes.data_as(pointer))
        return [red, blue]


@lru_cache(maxsize=1)
def native_lookup(path):
    return NativeColorLookup(path)
