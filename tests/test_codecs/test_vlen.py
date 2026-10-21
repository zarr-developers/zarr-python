from typing import Any

import numpy as np
import pytest

import zarr
from zarr import Array
from zarr.abc.codec import Codec, SupportsSyncCodec
from zarr.abc.store import Store
from zarr.codecs import ZstdCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.storage import StorePath

# The explicit id for the "str" literal avoids colliding with the auto-generated id for
# the `str` builtin.
numpy_str_dtypes: list[Any] = [
    None,
    str,
    pytest.param("str", id="str-literal"),
    np.dtypes.StrDType,
    "S",
    "U",
    np.dtypes.StringDType,
]
expected_array_string_dtype: np.dtype[Any] = np.dtypes.StringDType()


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
@pytest.mark.parametrize("dtype", numpy_str_dtypes)
@pytest.mark.parametrize("as_object_array", [False, True])
@pytest.mark.parametrize("compressor", [None, ZstdCodec()])
def test_vlen_string(
    store: Store, dtype: np.dtype[Any] | None, as_object_array: bool, compressor: Codec | None
) -> None:
    strings = ["hello", "world", "this", "is", "a", "test"]
    data = np.array(strings, dtype=dtype).reshape((2, 3))

    sp = StorePath(store, path="string")
    a = zarr.create_array(
        sp,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
        fill_value="",
        compressors=compressor,  # type: ignore[arg-type]
    )
    assert isinstance(a.metadata, ArrayV3Metadata)  # needed for mypy

    # should also work if input array is an object array, provided we explicitly specified
    # a stringlike dtype when creating the Array
    if as_object_array:
        data_obj = data.astype("O")

        a[:, :] = data_obj
    else:
        a[:, :] = data
    assert np.array_equal(data, a[:, :])
    assert a.metadata.data_type == get_data_type_from_native_dtype(data.dtype)
    assert a.dtype == data.dtype

    # test round trip
    b = Array.open(sp)
    assert isinstance(b.metadata, ArrayV3Metadata)  # needed for mypy
    assert np.array_equal(data, b[:, :])
    assert b.metadata.data_type == get_data_type_from_native_dtype(data.dtype)
    assert a.dtype == data.dtype


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@pytest.mark.parametrize("store", ["memory"], indirect=["store"])
@pytest.mark.parametrize(
    ("dtype", "fill_value", "elements"),
    [
        pytest.param("string", "", [f"S{i:05}" for i in range(9)], id="string"),
        pytest.param("variable_length_bytes", b"", [b"%05d" % i for i in range(9)], id="bytes"),
    ],
)
def test_vlen_f_contiguous(
    store: Store, dtype: str, fill_value: str | bytes, elements: list[str] | list[bytes]
) -> None:
    """An F-contiguous chunk written through a vlen codec round-trips in the original
    element order rather than the transposed memory order (gh-3558)."""
    # reshape(order="F") gives an F-contiguous view directly, so the whole-array write
    # below hands an F-contiguous chunk to the codec; assert it to guard the precondition.
    data = np.array(elements, dtype=object).reshape((3, 3), order="F")
    assert data.flags.f_contiguous
    sp = StorePath(store, path="vlen-f-contiguous")
    # chunks == shape so the write is a single complete chunk, which the codec pipeline
    # forwards to the codec untouched; a partial-chunk layout would be recopied to C order.
    a = zarr.create_array(
        sp, shape=data.shape, chunks=data.shape, dtype=dtype, fill_value=fill_value
    )
    a[:, :] = data
    assert np.array_equal(data, np.asarray(a[:, :], dtype=object))


def test_vlen_utf8_codec_supports_sync() -> None:
    assert isinstance(VLenUTF8Codec(), SupportsSyncCodec)


def test_vlen_bytes_codec_supports_sync() -> None:
    assert isinstance(VLenBytesCodec(), SupportsSyncCodec)
