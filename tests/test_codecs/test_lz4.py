from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.lz4 import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestLZ4Codec(BaseTestCodec):
    test_cls = _numcodecs.LZ4
    valid_json_v2 = ({"id": "lz4", "acceleration": 1},)
    valid_json_v3 = (
        {
            "name": "lz4",
            "configuration": {"acceleration": 1},
        },
        {
            "name": "numcodecs.lz4",
            "configuration": {"acceleration": 1},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)


def test_v3_json_alias() -> None:
    from zarr.codecs import numcodecs as _numcodecs

    """
    Test that the default JSON output of the legacy numcodecs.zarr3.LZ4 codec is readable, even if it's
    underspecified.
    """
    assert _numcodecs.LZ4.from_json(
        {"name": "numcodecs.lz4", "configuration": {}}
    ) == _numcodecs.LZ4(acceleration=1)
