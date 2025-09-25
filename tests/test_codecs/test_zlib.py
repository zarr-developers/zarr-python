from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.numcodecs.zlib import Zlib, check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestZlib(BaseTestCodec):
    test_cls = Zlib
    valid_json_v2 = ({"id": "zlib", "level": 1},)
    valid_json_v3 = (
        {
            "name": "zlib",
            "configuration": {"level": 1},
        },
        {
            "name": "numcodecs.zlib",
            "configuration": {"level": 1},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
