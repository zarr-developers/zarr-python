import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestZlibNumcodecsCodec(BaseTestCodec):
    test_cls = _numcodecs.Zlib
    valid_json_v2 = ({"id": "zlib", "level": 1},)  # type: ignore[typeddict-unknown-key]
    valid_json_v3 = (
        {
            "name": "zlib",
            "configuration": {"level": 1},
        },
    )
