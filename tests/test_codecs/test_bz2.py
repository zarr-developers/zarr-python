import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.bz2 import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestBZ2Codec(BaseTestCodec):
    test_cls = _numcodecs.BZ2
    valid_json_v2 = ({"id": "bz2", "level": 1},)
    valid_json_v3 = (
        {
            "name": "bz2",
            "configuration": {"level": 1},
        },
        {
            "name": "numcodecs.bz2",
            "configuration": {"level": 1},
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
    Test that the default JSON output of the legacy numcodecs.zarr3.BZ2 codec is readable, even if it's
    underspecified.
    """
    assert _numcodecs.BZ2.from_json(
        {"name": "numcodecs.bz2", "configuration": {}}
    ) == _numcodecs.BZ2(level=1)

    assert _numcodecs.BZ2.from_json({"name": "bz2", "configuration": {}}) == _numcodecs.BZ2(level=1)
