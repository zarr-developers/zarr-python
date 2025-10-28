import pytest

from tests.test_codecs.test_crc32c import TestCrc32cCodec
from zarr.codecs import numcodecs as _numcodecs

from .conftest import numcodecs_crc32c_available


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrDeprecationWarning")
@pytest.mark.skipif(
    not numcodecs_crc32c_available, reason="numcodecs crc32c codec is not available"
)
class TestCRC32CCodec(TestCrc32cCodec):
    test_cls = _numcodecs.CRC32C  # type: ignore[assignment]
    valid_json_v2 = (  # type: ignore[assignment]
        {"id": "crc32c"},
        {"id": "numcodecs.crc32c"},
        {"id": "crc32c", "location": "start"},
        {"id": "crc32c", "location": "end"},
    )

    @staticmethod
    def check_json_v3(data: object) -> bool:
        from zarr.codecs.numcodecs.crc32c import check_json_v3

        return check_json_v3(data)
