from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.crc32 import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestCRC32Codec(BaseTestCodec):
    test_cls = _numcodecs.CRC32
    valid_json_v2 = ({"id": "crc32"},)
    valid_json_v3 = (
        {"name": "crc32", "configuration": {}},
        {"name": "numcodecs.crc32", "configuration": {}},
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
