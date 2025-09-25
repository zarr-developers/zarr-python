import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestGZipNumcodecsCodec(BaseTestCodec):
    test_cls = _numcodecs.GZip
    valid_json_v2 = ({"id": "gzip", "level": 1},)
    valid_json_v3 = (
        {
            "name": "gzip",
            "configuration": {"level": 1},
        },
    )
