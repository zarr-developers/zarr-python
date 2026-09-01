import numpy as np
import pytest

from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.codecs import BloscCodec, BytesCodec, Crc32cCodec, GzipCodec, ZstdCodec
from zarr.metadata import DataType


@pytest.fixture
def buffer_prototype():
    return BufferPrototype(buffer=Buffer, nd_buffer=NDBuffer)


@pytest.fixture(params=list(DataType.__members__))
def dtype(request):
    return DataType[request.param]


# shape (0,) doesn't work with all blosc codecs; is it an important case to test?
@pytest.fixture(params=[(1,), (10,), (10, 5), (10, 1), (1, 10), (5, 6, 7), (1, 2, 3, 4, 5)])
def shape(request):
    return request.param


@pytest.fixture
def array_spec(buffer_prototype, dtype, shape):
    return ArraySpec(
        shape=shape,
        dtype=dtype.to_numpy_shortname(),
        fill_value=0,
        order="C",
        prototype=buffer_prototype,
    )


# TODO: parametrize all options
@pytest.fixture
def array_bytes_codec():
    return BytesCodec()


@pytest.fixture(params=[0, 1, 2])
def input_chunks_and_specs(request, array_spec):
    num_chunks = request.param
    chunk_data = [
        np.full(
            shape=array_spec.shape, fill_value=n, dtype=array_spec.dtype, order=array_spec.order
        )
        for n in range(num_chunks)
    ]
    return [
        (array_spec.prototype.nd_buffer.from_ndarray_like(data), array_spec) for data in chunk_data
    ]


async def test_array_bytes_codecs(array_bytes_codec, input_chunks_and_specs):
    encoded = await array_bytes_codec.encode(input_chunks_and_specs)
    assert len(encoded) == len(input_chunks_and_specs)
    encoded_chunks_and_specs = [
        (data, spec) for data, (_, spec) in zip(encoded, input_chunks_and_specs, strict=False)
    ]
    decoded = await array_bytes_codec.decode(encoded_chunks_and_specs)
    assert len(decoded) == len(input_chunks_and_specs)
    assert all(
        [
            np.array_equal(data.as_numpy_array(), decoded_data.as_numpy_array())
            for (data, _), decoded_data in zip(input_chunks_and_specs, decoded, strict=False)
        ]
    )


@pytest.fixture
async def input_bytes_and_specs(input_chunks_and_specs):
    # transform ndbuffers to buffers via bytes codec
    bytes_codec = BytesCodec()
    encoded = await bytes_codec.encode(input_chunks_and_specs)
    encoded_chunks_and_specs = [
        (data, spec) for data, (_, spec) in zip(encoded, input_chunks_and_specs, strict=False)
    ]
    return encoded_chunks_and_specs


@pytest.fixture(
    params=[
        pytest.param((GzipCodec, {}), id="GzipDefaults"),
        pytest.param((GzipCodec, {"level": 2}), id="GzipLev2"),
        pytest.param((ZstdCodec, {}), id="ZstdDefaults"),
        pytest.param((ZstdCodec, {"level": 2}), id="ZstdLev2"),
        pytest.param((ZstdCodec, {"level": 2, "checksum": True}), id="ZstdLev2Chksum"),
        pytest.param((Crc32cCodec, {}), id="Crc32c"),
    ]
)
def bytes_bytes_codec(request):
    Codec, kwargs = request.param
    return Codec(**kwargs)


async def test_bytes_bytes_codecs(bytes_bytes_codec, input_bytes_and_specs):
    encoded = await bytes_bytes_codec.encode(input_bytes_and_specs)
    assert len(encoded) == len(input_bytes_and_specs)
    encoded_bytes_and_specs = [
        (data, spec) for data, (_, spec) in zip(encoded, input_bytes_and_specs, strict=False)
    ]
    decoded = await bytes_bytes_codec.decode(encoded_bytes_and_specs)
    assert len(decoded) == len(input_bytes_and_specs)
    assert all(
        [
            np.array_equal(data.as_numpy_array(), decoded_data.as_numpy_array())
            for (data, _), decoded_data in zip(input_bytes_and_specs, decoded, strict=False)
        ]
    )


# blosc gets its own test because it has so many options
@pytest.mark.parametrize("shuffle", ["noshuffle", "shuffle", "bitshuffle"])
# "snappy" not supported by blosc (even though it's in the enum of options
@pytest.mark.parametrize("cname", ["lz4", "lz4hc", "blosclz", "zstd", "zlib"])
@pytest.mark.parametrize("clevel", [0, 3, 8])
async def test_blosc_codec(input_bytes_and_specs, shuffle, cname, clevel, shape):
    bytes_bytes_codec = BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)
    encoded = await bytes_bytes_codec.encode(input_bytes_and_specs)
    assert len(encoded) == len(input_bytes_and_specs)
    encoded_bytes_and_specs = [
        (data, spec) for data, (_, spec) in zip(encoded, input_bytes_and_specs, strict=False)
    ]
    decoded = await bytes_bytes_codec.decode(encoded_bytes_and_specs)
    assert len(decoded) == len(input_bytes_and_specs)
    assert all(
        [
            np.array_equal(data.as_numpy_array(), decoded_data.as_numpy_array())
            for (data, _), decoded_data in zip(input_bytes_and_specs, decoded, strict=False)
        ]
    )
