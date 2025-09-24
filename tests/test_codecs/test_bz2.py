import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestBZ2Codec(BaseTestCodec):
    test_cls = _numcodecs.BZ2
    valid_json_v2 = ({"id": "bz2", "level": 1},)
    valid_json_v3 = (
        {
            "name": "bz2",
            "configuration": {"level": 1},
        },
    )
