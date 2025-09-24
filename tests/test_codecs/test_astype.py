import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestAsTypeCodec(BaseTestCodec):
    test_cls = _numcodecs.AsType
    valid_json_v2 = ({"id": "astype", "encode_dtype": "<f4", "decode_dtype": "<f4"},)  # type: ignore[typeddict-unknown-key]
    valid_json_v3 = (
        {
            "name": "astype",
            "configuration": {"encode_dtype": "<f4", "decode_dtype": "<f4"},
        },
    )
