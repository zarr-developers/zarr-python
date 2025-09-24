import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestZstdNumcodecsCodec(BaseTestCodec):
    test_cls = _numcodecs.Zstd
    valid_json_v2 = ({"id": "zstd", "level": 0, "checksum": False},)  # type: ignore[typeddict-unknown-key]
    valid_json_v3 = (
        {
            "name": "zstd",
            "configuration": {"level": 0, "checksum": False},
        },
    )
