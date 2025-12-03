import io

import numpy as np
import pytest
from arro3.io import read_ipc_stream

import zarr
from zarr.codecs.arrow import ArrowIPCCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import numpy_buffer_prototype
from zarr.dtype import parse_dtype

CPU_BUFFER_PROTOTYPE = numpy_buffer_prototype()


numpy_array_fixtures = [
    (np.array([[1, 2, 3], [4, 5, 6]], dtype="int64"), None),
    (np.array([[1.5, 2.5], [3.5, 4.5]], dtype="float32"), None),
    (np.array([[True, False, True], [False, True, False]], dtype="bool"), None),
    (
        np.array(["foo", "barry", "bazo"], dtype=np.dtypes.StringDType()),
        zarr.dtype.VariableLengthUTF8(),
    ),
    # both come back as object dtype, but if we pass object array to Zarr, it complains about dtype resolution
    # np.array(['foo', 'barry', 'bazo'], dtype='U5'),
    # np.array(['foo', 'barry', 'bazo'], dtype=np.dtypes.StringDType())
]


@pytest.mark.parametrize("numpy_array_and_zdtype", numpy_array_fixtures)
async def test_arrow_codec_round_trip(numpy_array_and_zdtype) -> None:
    numpy_array, zdtype = numpy_array_and_zdtype
    if zdtype is None:
        spec_dtype = parse_dtype(numpy_array.dtype, zarr_format=3)
    else:
        spec_dtype = zdtype
    codec = ArrowIPCCodec()
    array_config = ArrayConfig(order="C", write_empty_chunks=True)
    array_spec = ArraySpec(
        shape=numpy_array.shape,
        dtype=spec_dtype,
        fill_value=0,
        config=array_config,
        prototype=CPU_BUFFER_PROTOTYPE,
    )

    ndbuffer = CPU_BUFFER_PROTOTYPE.nd_buffer.from_numpy_array(numpy_array)
    encoded = await codec._encode_single(ndbuffer, array_spec)
    decoded = await codec._decode_single(encoded, array_spec)

    # Test that the decoded array matches the original
    numpy_array_decoded = decoded.as_ndarray_like()
    np.testing.assert_array_equal(numpy_array_decoded, numpy_array)


async def test_custom_field_name() -> None:
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")
    spec_dtype = parse_dtype(numpy_array.dtype, zarr_format=3)
    codec = ArrowIPCCodec(column_name="custom_field_name")
    array_config = ArrayConfig(order="C", write_empty_chunks=True)
    array_spec = ArraySpec(
        shape=numpy_array.shape,
        dtype=spec_dtype,
        fill_value=0,
        config=array_config,
        prototype=CPU_BUFFER_PROTOTYPE,
    )

    ndbuffer = CPU_BUFFER_PROTOTYPE.nd_buffer.from_numpy_array(numpy_array)
    encoded = await codec._encode_single(ndbuffer, array_spec)
    decoded = await codec._decode_single(encoded, array_spec)

    # Test that the decoded array matches the original
    numpy_array_decoded = decoded.as_ndarray_like()
    np.testing.assert_array_equal(numpy_array_decoded, numpy_array)

    # test that we can read the arrow data directly
    record_batch_reader = read_ipc_stream(io.BytesIO(encoded.as_buffer_like()))
    record_batch = record_batch_reader.read_next_batch()
    assert record_batch.num_columns == 1
    _ = record_batch.column("custom_field_name")


def test_string_array() -> None:
    # IMO codec tests should be much more self contained,
    # not end-to-end array round-tripping tests.
    # But don't see a better way to test this at the moment..

    a = zarr.create_array(
        shape=4,
        chunks=2,
        dtype=zarr.dtype.VariableLengthUTF8(),
        serializer=ArrowIPCCodec(),
        store=zarr.storage.MemoryStore(),
    )

    a[:] = np.array(["abc", "1234", "foo", "bar"])
    result = a[:]
    np.testing.assert_equal(a, result)
