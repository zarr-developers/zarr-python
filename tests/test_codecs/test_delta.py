from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.numcodecs import delta as _numcodecs
from zarr.codecs.numcodecs.delta import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestDeltaCodec(BaseTestCodec):
    test_cls = _numcodecs.Delta
    valid_json_v2 = ({"id": "delta", "dtype": "|u1", "astype": "|u1"},)
    valid_json_v3 = (
        {
            "name": "delta",
            "configuration": {"dtype": "uint16", "astype": "uint8"},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
