import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestLZMACodec(BaseTestCodec):
    test_cls = _numcodecs.LZMA
    valid_json_v2 = ({"id": "lzma", "filters": None, "preset": None, "format": 1, "check": -1},)
    valid_json_v3 = (
        {
            "name": "lzma",
            "configuration": {"filters": None, "preset": None, "format": 1, "check": -1},
        },
    )
