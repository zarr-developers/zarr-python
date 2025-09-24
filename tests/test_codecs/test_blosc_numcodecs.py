import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestBloscNumcodecsCodec(BaseTestCodec):
    test_cls = _numcodecs.Blosc
    valid_json_v2 = ({"id": "blosc", "clevel": 5, "shuffle": 1, "blocksize": 0, "cname": "lz4"},)  # type: ignore[typeddict-unknown-key]
    valid_json_v3 = (
        {
            "name": "blosc",
            "configuration": {"clevel": 5, "shuffle": 1, "blocksize": 0, "cname": "lz4"},
        },
    )
