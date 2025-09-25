from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from tests.test_codecs.conftest import BaseTestCodec

from zarr import Array
from zarr.codecs import ZstdCodec
from zarr.codecs.vlen_utf8 import VLenBytesCodec, VLenUTF8Codec, VLenUTF8JSON_V2, VLenUTF8JSON_V3
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.dtype.npy.string import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.abc.codec import Codec
    from zarr.abc.store import Store


class TestVLenBytesCodec(BaseTestCodec):
    test_cls = VLenBytesCodec
    valid_json_v2 = ({"id": "vlen-bytes"},)
    valid_json_v3 = ({"name": "vlen-bytes"},"vlen-bytes")

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return VLenBytesCodec._check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return VLenBytesCodec._check_json_v3(data)

def test_vlen_utf8_to_json() -> None:
    codec = VLenUTF8Codec()
    expected_v2: VLenUTF8JSON_V2 = {"id": "vlen-utf8"}
    expected_v3: VLenUTF8JSON_V3 = {
        "name": "vlen-utf8",
    }
    assert codec.to_json(zarr_format=2) == expected_v2
    assert codec.to_json(zarr_format=3) == expected_v3