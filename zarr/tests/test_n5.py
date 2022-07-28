
import pytest

from zarr.n5 import N5ChunkWrapper
from numcodecs import GZip
import numpy as np


def test_make_n5_chunk_wrapper():
    dtype = 'uint8'
    chunk_shape = (10,)
    codec = GZip()
    # ValueError when specifying both compressor and compressor_config
    with pytest.raises(ValueError):
        N5ChunkWrapper(dtype,
                       chunk_shape=chunk_shape,
                       compressor_config=codec.get_config(),
                       compressor=codec)

    wrapper_a = N5ChunkWrapper(dtype, chunk_shape=chunk_shape, compressor_config=codec.get_config())
    wrapper_b = N5ChunkWrapper(dtype, chunk_shape=chunk_shape, compressor=codec)
    assert wrapper_a == wrapper_b


def test_partial_chunk_decode():
    dtype = 'uint8'
    chunk_shape = (4,)
    codec = GZip()
    codec_wrapped = N5ChunkWrapper(dtype, chunk_shape=chunk_shape, compressor=codec)
    data = np.zeros(chunk_shape, dtype=dtype)
    assert codec_wrapped.decode(codec_wrapped.encode(data[:2])) == data.tobytes()
