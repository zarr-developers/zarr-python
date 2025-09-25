import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestShuffleNumcodecsCodec(BaseTestCodec):
    test_cls = _numcodecs.Shuffle
    valid_json_v2 = ({"id": "shuffle", "elementsize": 4},)
    valid_json_v3 = (
        {
            "name": "shuffle",
            "configuration": {"elementsize": 4},
        },
    )
