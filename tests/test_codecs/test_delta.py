import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestDeltaCodec(BaseTestCodec):
    test_cls = _numcodecs.Delta
    valid_json_v2 = ({"id": "delta", "dtype": "|u1", "astype": "|u1"},)  # type: ignore[typeddict-unknown-key]
    valid_json_v3 = (
        {
            "name": "delta",
            "configuration": {"dtype": "|u1", "astype": "|u1"},
        },
    )
