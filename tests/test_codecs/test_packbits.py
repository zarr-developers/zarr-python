import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestPackBitsCodec(BaseTestCodec):
    test_cls = _numcodecs.PackBits
    valid_json_v2 = ({"id": "packbits"},)
    valid_json_v3 = ({"name": "packbits", "configuration": {}},)
