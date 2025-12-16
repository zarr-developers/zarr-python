from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.fletcher32 import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestFletcher32Codec(BaseTestCodec):
    test_cls = _numcodecs.Fletcher32
    valid_json_v2 = ({"id": "fletcher32"},)
    valid_json_v3 = (
        {"name": "fletcher32", "configuration": {}},
        {"name": "numcodecs.fletcher32", "configuration": {}},
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
    Test that the default JSON output of the legacy numcodecs.zarr3.Fletcher32 codec is readable, even if it's
    underspecified.
    """
    assert (
        _numcodecs.Fletcher32.from_json({"name": "numcodecs.fletcher32", "configuration": {}})
        == _numcodecs.Fletcher32()
    )
