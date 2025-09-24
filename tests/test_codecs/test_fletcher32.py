import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestFletcher32Codec(BaseTestCodec):
    test_cls = _numcodecs.Fletcher32
    valid_json_v2 = ({"id": "fletcher32"},)
    valid_json_v3 = ({"name": "fletcher32", "configuration": {}},)
