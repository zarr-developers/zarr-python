import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestAdler32Codec(BaseTestCodec):
    test_cls = _numcodecs.Adler32
    valid_json_v2 = ({"id": "adler32"},)
    valid_json_v3 = ({"name": "adler32", "configuration": {}},)
