import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestCRC32CCodec(BaseTestCodec):
    test_cls = _numcodecs.CRC32C
    valid_json_v2 = ({"id": "crc32c"},)
    valid_json_v3 = ({"name": "crc32c", "configuration": {}},)
