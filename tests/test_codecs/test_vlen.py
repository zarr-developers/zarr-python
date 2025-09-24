from typing import Any

import numpy as np
import pytest

import zarr
from tests.test_codecs.conftest import BaseTestCodec
from zarr import Array
from zarr.abc.codec import Codec
from zarr.abc.store import Store
from zarr.codecs import ZstdCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec, VLenUTF8JSON_V2, VLenUTF8JSON_V3
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.storage import StorePath

numpy_str_dtypes: list[type | str | None] = [None, str, "str", np.dtypes.StrDType, "S", "U"]
expected_array_string_dtype: np.dtype[Any]
if _NUMPY_SUPPORTS_VLEN_STRING:
    numpy_str_dtypes.append(np.dtypes.StringDType)
    expected_array_string_dtype = np.dtypes.StringDType()
else:
    expected_array_string_dtype = np.dtype("O")


class TestVLenUTF8Codec(BaseTestCodec):
    test_cls = VLenUTF8Codec
    valid_json_v2 = ({"id": "vlen-utf8"},)
    valid_json_v3 = ({"name": "vlen-utf8"},)


class TestVLenBytesCodec(BaseTestCodec):
    test_cls = VLenBytesCodec
    valid_json_v2 = ({"id": "vlen-bytes"},)
    valid_json_v3 = ({"name": "vlen-bytes"},)


def test_vlen_utf8_to_json() -> None:
    codec = VLenUTF8Codec()
    expected_v2: VLenUTF8JSON_V2 = {"id": "vlen-utf8"}
    expected_v3: VLenUTF8JSON_V3 = {
        "name": "vlen-utf8",
    }
    assert codec.to_json(zarr_format=2) == expected_v2
    assert codec.to_json(zarr_format=3) == expected_v3


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
