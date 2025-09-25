from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.bitround import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestBitRoundCodec(BaseTestCodec):
    test_cls = _numcodecs.BitRound
    valid_json_v2 = ({"id": "bitround", "keepbits": 8},)
    valid_json_v3 = (
        {
            "name": "bitround",
            "configuration": {"keepbits": 8},
        },
        {
            "name": "numcodecs.bitround",
            "configuration": {"keepbits": 8},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
