from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.lzma import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestLZMACodec(BaseTestCodec):
    test_cls = _numcodecs.LZMA
    valid_json_v2 = ({"id": "lzma", "filters": None, "preset": None, "format": 1, "check": -1},)
    valid_json_v3 = (
        {
            "name": "lzma",
            "configuration": {"filters": None, "preset": None, "format": 1, "check": -1},
        },
        {
            "name": "numcodecs.lzma",
            "configuration": {"filters": None, "preset": None, "format": 1, "check": -1},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
