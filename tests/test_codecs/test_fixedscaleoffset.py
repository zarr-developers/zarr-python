import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestFixedScaleOffsetCodec(BaseTestCodec):
    test_cls = _numcodecs.FixedScaleOffset
    valid_json_v2 = (
        {"id": "fixedscaleoffset", "scale": 1.0, "offset": 0.0, "astype": "|u1", "dtype": "|u1"},  # type: ignore[typeddict-unknown-key]
    )
    valid_json_v3 = (
        {
            "name": "fixedscaleoffset",
            "configuration": {"scale": 1.0, "offset": 0.0, "astype": "|u1", "dtype": "|u1"},
        },
    )
