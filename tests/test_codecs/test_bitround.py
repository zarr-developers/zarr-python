import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestBitRoundCodec(BaseTestCodec):
    test_cls = _numcodecs.BitRound
    valid_json_v2 = ({"id": "bitround", "keepbits": 8},)
    valid_json_v3 = (
        {
            "name": "bitround",
            "configuration": {"keepbits": 8},
        },
    )
