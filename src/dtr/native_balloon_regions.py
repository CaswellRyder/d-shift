"""Checked research-only interface for hull-preserving MSER reduction."""

import ctypes as ct
from pathlib import Path

import numpy as np

from .data import read_json, sha256


class NativeBalloonRegions:
    def __init__(self, path, direct=False):
        if type(direct) is not bool:
            raise ValueError("Direct-pointer mode must be explicit boolean")
        self.direct = direct
        path = Path(path).resolve()
        receipt = read_json(path.with_suffix(".json"))
        if receipt.get("contract") != "dtr-balloon-regions-v1" or receipt.get("sha256") != sha256(
            path
        ):
            raise ValueError("Native region contract/hash mismatch")
        self.path = path
        self.lib = ct.CDLL(str(path))
        self.lib.dtr_balloon_regions_abi.argtypes = []
        self.lib.dtr_balloon_regions_abi.restype = ct.c_uint32
        if self.lib.dtr_balloon_regions_abi() != 1:
            raise ValueError("Native region ABI mismatch")
        self.lib.dtr_balloon_region.argtypes = [
            ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_int32),
            ct.c_size_t,
            ct.POINTER(ct.c_int32),
        ]
        self.lib.dtr_balloon_region.restype = ct.c_int32
        if direct:
            # Buffers remain owned by the local arrays until this synchronous
            # call returns. Avoid three NumPy/ctypes cast wrappers per region.
            self.lib.dtr_balloon_region.argtypes = [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_size_t,
                ct.c_void_p,
            ]

    def reduce(self, plane, points):
        if plane.dtype != np.uint8 or plane.shape != (240, 320) or not plane.flags.c_contiguous:
            raise ValueError("Expected contiguous 320x240 uint8 plane")
        if (
            points.dtype != np.int32
            or points.ndim != 2
            or points.shape[1] != 2
            or not points.flags.c_contiguous
            or not 1 <= len(points) <= 76800
        ):
            raise ValueError("Expected contiguous nonempty Nx2 int32 points")
        out = np.empty((480, 2), dtype=np.int32)
        if self.direct:
            n = self.lib.dtr_balloon_region(
                plane.ctypes.data,
                points.ctypes.data,
                len(points),
                out.ctypes.data,
            )
        else:
            n = self.lib.dtr_balloon_region(
                plane.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                points.ctypes.data_as(ct.POINTER(ct.c_int32)),
                len(points),
                out.ctypes.data_as(ct.POINTER(ct.c_int32)),
            )
        if not 0 <= n <= 480:
            raise ValueError("Invalid region coordinates or native result")
        return out[:n]
