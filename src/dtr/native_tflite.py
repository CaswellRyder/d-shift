"""Small INT8/FP32 adapter for the system TensorFlow Lite C API.

Uses the existing shared library, not TensorFlow or Python-version-specific wheels.
API contract: tensorflow/lite/core/c/c_api.h. One input and one output only.
"""

import ctypes as ct
from ctypes.util import find_library
import os
from pathlib import Path

import numpy as np


class Quantization(ct.Structure):
    _fields_ = [("scale", ct.c_float), ("zero_point", ct.c_int32)]


class Interpreter:
    def __init__(self, model_path, num_threads=1):
        candidate = os.environ.get("DTR_TFLITE_LIBRARY") or find_library("tensorflow-lite")
        if not candidate:
            candidate = "/usr/lib/arm-linux-gnueabihf/libtensorflow-lite.so.2.20.0"
        self.lib = ct.CDLL(candidate)
        self._model = self._options = self._interpreter = None
        definitions = {
            "TfLiteVersion": (ct.c_char_p, []),
            "TfLiteModelCreateFromFile": (ct.c_void_p, [ct.c_char_p]),
            "TfLiteModelDelete": (None, [ct.c_void_p]),
            "TfLiteInterpreterOptionsCreate": (ct.c_void_p, []),
            "TfLiteInterpreterOptionsDelete": (None, [ct.c_void_p]),
            "TfLiteInterpreterOptionsSetNumThreads": (None, [ct.c_void_p, ct.c_int32]),
            "TfLiteInterpreterCreate": (ct.c_void_p, [ct.c_void_p, ct.c_void_p]),
            "TfLiteInterpreterDelete": (None, [ct.c_void_p]),
            "TfLiteInterpreterAllocateTensors": (ct.c_int, [ct.c_void_p]),
            "TfLiteInterpreterInvoke": (ct.c_int, [ct.c_void_p]),
            "TfLiteInterpreterGetInputTensorCount": (ct.c_int32, [ct.c_void_p]),
            "TfLiteInterpreterGetOutputTensorCount": (ct.c_int32, [ct.c_void_p]),
            "TfLiteInterpreterGetInputTensor": (ct.c_void_p, [ct.c_void_p, ct.c_int32]),
            "TfLiteInterpreterGetOutputTensor": (ct.c_void_p, [ct.c_void_p, ct.c_int32]),
            "TfLiteTensorType": (ct.c_int, [ct.c_void_p]),
            "TfLiteTensorNumDims": (ct.c_int32, [ct.c_void_p]),
            "TfLiteTensorDim": (ct.c_int32, [ct.c_void_p, ct.c_int32]),
            "TfLiteTensorByteSize": (ct.c_size_t, [ct.c_void_p]),
            "TfLiteTensorQuantizationParams": (Quantization, [ct.c_void_p]),
            "TfLiteTensorCopyFromBuffer": (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_size_t]),
            "TfLiteTensorCopyToBuffer": (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_size_t]),
        }
        for name, (result, arguments) in definitions.items():
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = result, arguments
        self.version = self.lib.TfLiteVersion().decode()
        try:
            self._model = self.lib.TfLiteModelCreateFromFile(os.fsencode(Path(model_path)))
            self._options = self.lib.TfLiteInterpreterOptionsCreate()
            if not self._model or not self._options:
                raise RuntimeError("Native TFLite could not load model/options")
            self.lib.TfLiteInterpreterOptionsSetNumThreads(self._options, num_threads)
            self._interpreter = self.lib.TfLiteInterpreterCreate(self._model, self._options)
            if not self._interpreter:
                raise RuntimeError("Native TFLite could not create interpreter")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f"Native TFLite operation failed: {status}")

    def allocate_tensors(self):
        self._check(self.lib.TfLiteInterpreterAllocateTensors(self._interpreter))
        if (self.lib.TfLiteInterpreterGetInputTensorCount(self._interpreter) != 1
                or self.lib.TfLiteInterpreterGetOutputTensorCount(self._interpreter) != 1):
            raise ValueError("Native adapter requires exactly one input and one output")
        self._tensors = [self.lib.TfLiteInterpreterGetInputTensor(self._interpreter, 0),
                         self.lib.TfLiteInterpreterGetOutputTensor(self._interpreter, 0)]
        self._details = []
        for index, tensor in enumerate(self._tensors):
            kind = self.lib.TfLiteTensorType(tensor) if tensor else None
            if kind not in (9, 1):  # kTfLiteInt8, kTfLiteFloat32
                raise ValueError("Native adapter supports INT8 or FP32 tensors only")
            dtype = np.int8 if kind == 9 else np.float32
            rank = self.lib.TfLiteTensorNumDims(tensor)
            if not 1 <= rank <= 4:
                raise ValueError("Invalid tensor rank")
            shape = np.array([self.lib.TfLiteTensorDim(tensor, d) for d in range(rank)])
            quant = self.lib.TfLiteTensorQuantizationParams(tensor)
            if (np.any(shape <= 0) or int(np.prod(shape))*np.dtype(dtype).itemsize != self.lib.TfLiteTensorByteSize(tensor)
                    or (kind == 9 and (not np.isfinite(quant.scale) or quant.scale <= 0))):
                raise ValueError("Invalid tensor shape/quantization")
            self._details.append(dict(index=index, shape=shape, dtype=dtype,
                                      quantization=(quant.scale, quant.zero_point)))

    def get_input_details(self):
        return [self._details[0]]

    def get_output_details(self):
        return [self._details[1]]

    def set_tensor(self, index, values):
        if index != 0 or values.dtype != self._details[0]["dtype"] or list(values.shape) != list(self._details[0]["shape"]):
            raise ValueError("Native input shape/type/index mismatch")
        values = np.ascontiguousarray(values)
        self._check(self.lib.TfLiteTensorCopyFromBuffer(self._tensors[0], values.ctypes.data, values.nbytes))

    def invoke(self):
        self._check(self.lib.TfLiteInterpreterInvoke(self._interpreter))

    def get_tensor(self, index):
        if index != 1:
            raise ValueError("Invalid native output index")
        values = np.empty(self._details[1]["shape"], dtype=self._details[1]["dtype"])
        self._check(self.lib.TfLiteTensorCopyToBuffer(self._tensors[1], values.ctypes.data, values.nbytes))
        return values

    def close(self):
        for field, deleter in [("_interpreter", "TfLiteInterpreterDelete"),
                               ("_options", "TfLiteInterpreterOptionsDelete"),
                               ("_model", "TfLiteModelDelete")]:
            pointer = getattr(self, field, None)
            if pointer:
                getattr(self.lib, deleter)(pointer)
                setattr(self, field, None)

    def __del__(self):
        self.close()
