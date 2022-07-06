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


cupy = pytest.importorskip("cupy")


class CuPyCPUCompressor(Codec):
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


@pytest.mark.parametrize("compressor", [None, "Zlib", "Blosc"])
@pytest.mark.parametrize("store_type", [None, DirectoryStore, MemoryStore, ZipStore])
def test_array(tmp_path, compressor, store_type):
    compressor = init_compressor(compressor)

    # with cupy array
    store = init_store(tmp_path / "from_cupy_array", store_type)
    a = cupy.arange(100)
    z = array(
        a, chunks=10, compressor=compressor, store=store, meta_array=cupy.empty(())
    )
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert isinstance(a, type(z[:]))
    cupy.testing.assert_array_equal(a, z[:])

    # with array-like
    store = init_store(tmp_path / "from_list", store_type)
    a = list(range(100))
    z = array(
        a, chunks=10, compressor=compressor, store=store, meta_array=cupy.empty(())
    )
    assert (100,) == z.shape
    assert np.asarray(a).dtype == z.dtype
    cupy.testing.assert_array_equal(a, z[:])

    # with another zarr array
    store = init_store(tmp_path / "from_another_store", store_type)
    z2 = array(z, compressor=compressor, store=store, meta_array=cupy.empty(()))
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    cupy.testing.assert_array_equal(z[:], z2[:])


@pytest.mark.parametrize("compressor", [None, "Zlib", "Blosc"])
def test_empty(compressor):
    z = empty(
        100,
        chunks=10,
        compressor=init_compressor(compressor),
        meta_array=cupy.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks


@pytest.mark.parametrize("compressor", [None, "Zlib", "Blosc"])
def test_zeros(compressor):
    z = zeros(
        100,
        chunks=10,
        compressor=init_compressor(compressor),
        meta_array=cupy.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks
    cupy.testing.assert_array_equal(np.zeros(100), z[:])


@pytest.mark.parametrize("compressor", [None, "Zlib", "Blosc"])
def test_ones(compressor):
    z = ones(
        100,
        chunks=10,
        compressor=init_compressor(compressor),
        meta_array=cupy.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks
    cupy.testing.assert_array_equal(np.ones(100), z[:])


@pytest.mark.parametrize("compressor", [None, "Zlib", "Blosc"])
def test_full(compressor):
    z = full(
        100,
        chunks=10,
        fill_value=42,
        dtype="i4",
        compressor=init_compressor(compressor),
        meta_array=cupy.empty(()),
    )
    assert (100,) == z.shape
    assert (10,) == z.chunks
    cupy.testing.assert_array_equal(np.full(100, fill_value=42, dtype="i4"), z[:])

    # nan
    z = full(
        100,
        chunks=10,
        fill_value=np.nan,
        dtype="f8",
        compressor=init_compressor(compressor),
        meta_array=cupy.empty(()),
    )
    assert np.all(np.isnan(z[:]))


@pytest.mark.parametrize("compressor", [None, "Zlib", "Blosc"])
@pytest.mark.parametrize("store_type", [None, DirectoryStore, MemoryStore, ZipStore])
def test_group(tmp_path, compressor, store_type):
    store = init_store(tmp_path, store_type)
    g = open_group(store, meta_array=cupy.empty(()))
    g.ones("data", shape=(10, 11), dtype=int, compressor=init_compressor(compressor))
    a = g["data"]
    assert a.shape == (10, 11)
    assert a.dtype == int
    assert isinstance(a, Array)
    assert isinstance(a[:], cupy.ndarray)
    assert (a[:] == 1).all()
