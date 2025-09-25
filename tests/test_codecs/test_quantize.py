from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.quantize import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestQuantizeCodec(BaseTestCodec):
    test_cls = _numcodecs.Quantize
    valid_json_v2 = ({"id": "quantize", "digits": 3, "astype": "<f4", "dtype": "<f4"},)
    valid_json_v3 = (
        {
            "name": "quantize",
            "configuration": {"digits": 3, "astype": "float32", "dtype": "float32"},
        },
        {
            "name": "numcodecs.quantize",
            "configuration": {"digits": 3, "astype": "<f4", "dtype": "<f4"},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
