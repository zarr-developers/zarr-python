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
