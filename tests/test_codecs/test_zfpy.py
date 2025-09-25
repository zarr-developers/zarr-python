import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs

pytest.importorskip("zfpy")


class TestZFPYCodec(BaseTestCodec):
    test_cls = _numcodecs.ZFPY
    valid_json_v2 = (
        {
            "id": "zfpy",
            "mode": 4,
            "compression_kwargs": {"tolerance": -1},
            "precision": -1,
            "rate": -1,
            "tolerance": -1,
        },
    )
    valid_json_v3 = (
        {
            "name": "zfpy",
            "configuration": {
                "mode": 4,
                "compression_kwargs": {"tolerance": -1},
                "precision": -1,
                "rate": -1,
                "tolerance": -1,
            },
        },
    )
