import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.numcodecs.adler32 import Adler32, check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestAdler32Codec(BaseTestCodec):
    test_cls = Adler32
    valid_json_v2 = (
        {"id": "adler32"},
        {"id": "adler32", "location": "start"},
        {"id": "adler32", "location": "end"},
    )
    valid_json_v3 = (
        {"name": "adler32", "configuration": {}},
        {"name": "adler32", "configuration": {"location": "start"}},
        {"name": "adler32", "configuration": {"location": "end"}},
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
    Test that the default JSON output of the legacy numcodecs.zarr3.Adler32 codec is readable, even if it's
    underspecified.
    """
    assert (
        _numcodecs.Adler32.from_json({"name": "numcodecs.adler32", "configuration": {}})
        == _numcodecs.Adler32()
    )
