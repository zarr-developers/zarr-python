"""The wrapper defers selection; explicit execution touches the source."""

from typing import Any

import numpy as np
import pytest

from zarr_indexing import EagerArrayAdapter, LazyArray


class RecordingArray:
    def __init__(self) -> None:
        self.data = np.arange(30).reshape(5, 6)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.reads: list[Any] = []

    def __getitem__(self, key: Any) -> Any:
        self.reads.append(key)
        return self.data[key]


@pytest.mark.parametrize("mode", ["basic", "oindex", "vindex", "iteration"])
def test_selection_defers_reads(mode: str) -> None:
    source = RecordingArray()
    array = LazyArray(source)
    if mode == "basic":
        view = array[1:][::-1, ::2]
        expected = source.data[1:][::-1, ::2]
    elif mode == "oindex":
        view = array.oindex[[4, 1], :][:, ::2]
        expected = source.data[[4, 1]][:, ::2]
    elif mode == "vindex":
        view = array.vindex[[4, 1], [2, 0]][::-1]
        expected = source.data[[4, 1], [2, 0]][::-1]
    else:
        view = list(array)[2]
        expected = source.data[2]
    assert isinstance(view, LazyArray)
    assert source.reads == []
    np.testing.assert_array_equal(view.result(), expected)
    assert source.reads


def test_no_redundant_lazy_accessor() -> None:
    assert not hasattr(LazyArray(np.arange(3)), "lazy")


@pytest.mark.parametrize("selection", [slice(None), slice(0, 0), 0])
def test_eager_adapter_executes_selection(selection: Any) -> None:
    source = RecordingArray()
    view = LazyArray(source)[1:, ::2]
    adapter = EagerArrayAdapter(view)
    assert (adapter.shape, adapter.ndim, adapter.dtype) == (view.shape, view.ndim, view.dtype)
    assert source.reads == []
    result = adapter[selection]
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, source.data[1:, ::2][selection])
    assert not np.shares_memory(result, source.data)


def test_eager_adapter_distinguishes_token_semantics() -> None:
    view = LazyArray(np.arange(8))[::2]
    assert EagerArrayAdapter(view).__dask_tokenize__() != view.__dask_tokenize__()


def test_eager_adapter_refuses_no_copy_conversion() -> None:
    with pytest.raises(ValueError, match="copy"):
        EagerArrayAdapter(LazyArray(np.arange(3))).__array__(copy=False)
