import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.adler32 import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestAdler32Codec(BaseTestCodec):
    test_cls = _numcodecs.Adler32
    valid_json_v2 = (
        {"id": "adler32"},
        {"id": "adler32", "location": "start"},
        {"id": "adler32", "location": "end"},
    )
    valid_json_v3 = (
        {"name": "numcodecs.adler32", "configuration": {}},
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
