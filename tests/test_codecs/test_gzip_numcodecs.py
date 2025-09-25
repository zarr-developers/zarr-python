import pytest

from tests.test_codecs.test_gzip import TestGZipCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestGZipNumcodecsCodec(TestGZipCodec):
    test_cls = _numcodecs.GZip
    valid_json_v2 = ({"id": "gzip", "level": 1},)
    valid_json_v3 = (
        {
            "name": "gzip",
            "configuration": {"level": 1},
        },
        {
            "name": "numcodecs.gzip",
            "configuration": {"level": 1},
        },
    )
