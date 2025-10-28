import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.packbits import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestPackBitsCodec(BaseTestCodec):
    test_cls = _numcodecs.PackBits
    valid_json_v2 = ({"id": "packbits"},)
    valid_json_v3 = (
        {"name": "packbits", "configuration": {}},
        {"name": "numcodecs.packbits", "configuration": {}},
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
    Test that the default JSON output of the legacy numcodecs.zarr3.PackBits codec is readable, even if it's
    underspecified.
    """
    assert (
        _numcodecs.PackBits.from_json({"name": "numcodecs.packbits", "configuration": {}})
        == _numcodecs.PackBits()
    )
