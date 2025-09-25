import pytest

from tests.test_codecs.test_crc32c import TestCrc32cCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestCRC32CCodec(TestCrc32cCodec):
    test_cls = _numcodecs.CRC32C  # type: ignore[assignment]
    valid_json_v2 = (  # type: ignore[assignment]
        {"id": "crc32c"},
        {"id": "numcodecs.crc32c"},
        {"id": "crc32c", "location": "start"},
        {"id": "crc32c", "location": "end"},
    )
