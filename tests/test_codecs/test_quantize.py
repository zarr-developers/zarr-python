import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestQuantizeCodec(BaseTestCodec):
    test_cls = _numcodecs.Quantize
    valid_json_v2 = ({"id": "quantize", "digits": 3, "astype": "<f4", "dtype": "<f4"},)
    valid_json_v3 = (
        {
            "name": "quantize",
            "configuration": {"digits": 3, "astype": "<f4", "dtype": "<f4"},
        },
    )
