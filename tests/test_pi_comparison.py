import importlib.util
import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dtr.native_tflite import Interpreter

spec = importlib.util.spec_from_file_location(
    "pi_compare", Path(__file__).parents[1] / "scripts/pi_compare.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_localization_duplicates_and_misses():
    assert module.localization_counts([[0,0,10,10], [0,0,10,10]], [[0,0,10,10]]) == dict(tp=1, fp=1, fn=0)
    assert module.localization_counts([], [[0,0,10,10]]) == dict(tp=0, fp=0, fn=1)
    assert module.localization_counts([[0,0,10,10]], []) == dict(tp=0, fp=1, fn=0)


def test_localization_matching_is_not_prediction_order_dependent():
    truth = [[0,0,10,10], [3,0,13,10]]
    predictions = [[1,0,11,10], [0,0,7,10]]
    assert module.localization_counts(predictions, truth)["tp"] == 2
    assert module.localization_counts(predictions[::-1], truth)["tp"] == 2


def test_native_input_guard_prevents_wrong_buffer_copy():
    adapter = Interpreter.__new__(Interpreter)
    adapter._details = [dict(shape=[1,2,2,3], dtype=np.int8)]
    for index, values in [(1, np.zeros((1,2,2,3), np.int8)),
                          (0, np.zeros((1,2,2,3), np.float32)),
                          (0, np.zeros((1,1,1,3), np.int8))]:
        with pytest.raises(ValueError, match="mismatch"):
            adapter.set_tensor(index, values)


def test_native_status_checked():
    Interpreter._check(0)
    with pytest.raises(RuntimeError):
        Interpreter._check(1)


@pytest.mark.parametrize("dtype",[np.int8,np.float32])
def test_native_tensor_copy_respects_element_size(dtype):
    adapter = Interpreter.__new__(Interpreter)
    adapter._details = [dict(shape=[1,2],dtype=dtype),dict(shape=[1,2],dtype=dtype)]
    adapter._tensors = [123,456]
    values = np.array([[2,3]],dtype=dtype)
    copied = []

    def copy_from(tensor,pointer,size):
        copied.append((tensor,ctypes.string_at(pointer,size)))
        return 0

    def copy_to(tensor,pointer,size):
        assert tensor == 456 and size == values.nbytes
        ctypes.memmove(pointer,values.ctypes.data,size)
        return 0

    adapter.lib = SimpleNamespace(TfLiteTensorCopyFromBuffer=copy_from,TfLiteTensorCopyToBuffer=copy_to)
    adapter.set_tensor(0,values)
    assert copied == [(123,values.tobytes())]
    np.testing.assert_array_equal(adapter.get_tensor(1),values)
    with pytest.raises(ValueError,match="mismatch"):
        adapter.set_tensor(0,values.astype(np.float64))


def test_metrics_empty():
    assert module.metrics(dict(tp=0, fp=0, fn=0))["f1"] == 0
