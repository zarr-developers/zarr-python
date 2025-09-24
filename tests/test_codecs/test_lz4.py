import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestLZ4Codec(BaseTestCodec):
    test_cls = _numcodecs.LZ4
    valid_json_v2 = ({"id": "lz4", "acceleration": 1},)  # type: ignore[typeddict-unknown-key]
    valid_json_v3 = (
        {
            "name": "lz4",
            "configuration": {"acceleration": 1},
        },
    )
