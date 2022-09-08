from typing import Optional
import numpy as np
import pytest

from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray_like
from numcodecs.registry import get_codec, register_codec

import zarr.codecs
from zarr.core import Array
from zarr.creation import array, empty, full, ones, zeros
from zarr.hierarchy import open_group
from zarr.storage import DirectoryStore, MemoryStore, Store, ZipStore


class CuPyCPUCompressor(Codec):  # pragma: no cover
    """CPU compressor for CuPy arrays

    This compressor converts CuPy arrays host memory before compressing
    the arrays using `compressor`.

    Parameters
    ----------
    compressor : numcodecs.abc.Codec
        The codec to use for compression and decompression.
    """

    codec_id = "cupy_cpu_compressor"

    def __init__(self, compressor: Codec = None):
        self.compressor = compressor

    def encode(self, buf):
        import cupy

        buf = cupy.asnumpy(ensure_contiguous_ndarray_like(buf))
        if self.compressor:
            buf = self.compressor.encode(buf)
        return buf

    def decode(self, chunk, out=None):
        import cupy

        if self.compressor:
            cpu_out = None if out is None else cupy.asnumpy(out)
            chunk = self.compressor.decode(chunk, cpu_out)

        chunk = cupy.asarray(ensure_contiguous_ndarray_like(chunk))
        if out is not None:
            cupy.copyto(out, chunk.view(dtype=out.dtype), casting="no")
            chunk = out
        return chunk

    def get_config(self):
        cc_config = self.compressor.get_config() if self.compressor else None
        return {
            "id": self.codec_id,
            "compressor_config": cc_config,
        }

    @classmethod
    def from_config(cls, config):
        cc_config = config.get("compressor_config", None)
        compressor = get_codec(cc_config) if cc_config else None
        return cls(compressor=compressor)


register_codec(CuPyCPUCompressor)


class MyArray(np.ndarray):
    """Dummy array class to test the `meta_array` argument

    Useful when CuPy isn't available.

    This class also makes some of the functions from the numpy
    module available.
    """

    testing = np.testing

    @classmethod
    def arange(cls, size):
        ret = cls(shape=(size,), dtype="int64")
        ret[:] = range(size)
        return ret

    @classmethod
    def empty(cls, shape):
        return cls(shape=shape)


def init_compressor(compressor) -> CuPyCPUCompressor:
    if compressor:
        compressor = getattr(zarr.codecs, compressor)()
    return CuPyCPUCompressor(compressor)


def init_store(tmp_path, store_type) -> Optional[Store]:
    if store_type is DirectoryStore:
        return store_type(str(tmp_path / "store"))
    if store_type is MemoryStore:
        return MemoryStore()
    return None


def ensure_module(module):
    if isinstance(module, str):
        return pytest.importorskip(module)
    return module


param_module_and_compressor = [
    (MyArray, None),
    ("cupy", init_compressor(None)),
    ("cupy", init_compressor("Zlib")),
    ("cupy", init_compressor("Blosc")),
]


@pytest.mark.parametrize("module, compressor", param_module_and_compressor)
@pytest.mark.parametrize("store_type", [None, DirectoryStore, MemoryStore, ZipStore])
def test_array(tmp_path, module, compressor, store_type):
    xp = ensure_module(module)

    store = init_store(tmp_path / "from_cupy_array", store_type)
    a = xp.arange(100)
    z = array(a, chunks=10, compressor=compressor, store=store, meta_array=xp.empty(()))
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(a, type(z[:]))
    assert isinstance(z.meta_array, type(xp.empty(())))
    xp.testing.assert_array_equal(a, z[:])

    # with array-like
    store = init_store(tmp_path / "from_list", store_type)
    a = list(range(100))
    z = array(a, chunks=10, compressor=compressor, store=store, meta_array=xp.empty(()))
    assert (100,) == z.shape
    assert np.asarray(a).dtype == z.dtype
    xp.testing.assert_array_equal(a, z[:])

    # with another zarr array
    store = init_store(tmp_path / "from_another_store", store_type)
    z2 = array(z, compressor=compressor, store=store, meta_array=xp.empty(()))
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    xp.testing.assert_array_equal(z[:], z2[:])


@pytest.mark.parametrize("module, compressor", param_module_and_compressor)
def test_empty(module, compressor):
    xp = ensure_module(module)
    z = empty(
        100,
        chunks=10,
        compressor=compressor,
        meta_array=xp.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks


@pytest.mark.parametrize("module, compressor", param_module_and_compressor)
def test_zeros(module, compressor):
    xp = ensure_module(module)
    z = zeros(
        100,
        chunks=10,
        compressor=compressor,
        meta_array=xp.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks
    xp.testing.assert_array_equal(np.zeros(100), z[:])


@pytest.mark.parametrize("module, compressor", param_module_and_compressor)
def test_ones(module, compressor):
    xp = ensure_module(module)
    z = ones(
        100,
        chunks=10,
        compressor=compressor,
        meta_array=xp.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks
    xp.testing.assert_array_equal(np.ones(100), z[:])


@pytest.mark.parametrize("module, compressor", param_module_and_compressor)
def test_full(module, compressor):
    xp = ensure_module(module)
    z = full(
        100,
        chunks=10,
        fill_value=42,
        dtype="i4",
        compressor=compressor,
        meta_array=xp.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks
    xp.testing.assert_array_equal(np.full(100, fill_value=42, dtype="i4"), z[:])

    # nan
    z = full(
        100,
        chunks=10,
        fill_value=np.nan,
        dtype="f8",
        compressor=compressor,
        meta_array=xp.empty(()),
    )
    assert np.all(np.isnan(z[:]))


@pytest.mark.parametrize("module, compressor", param_module_and_compressor)
@pytest.mark.parametrize("store_type", [None, DirectoryStore, MemoryStore, ZipStore])
def test_group(tmp_path, module, compressor, store_type):
    xp = ensure_module(module)
    store = init_store(tmp_path, store_type)
    g = open_group(store, meta_array=xp.empty(()))
    g.ones("data", shape=(10, 11), dtype=int, compressor=compressor)
    a = g["data"]
    assert a.shape == (10, 11)
    assert a.dtype == int
    assert isinstance(a, Array)
    assert isinstance(a[:], type(xp.empty(())))
    assert (a[:] == 1).all()
    assert isinstance(g.meta_array, type(xp.empty(())))
