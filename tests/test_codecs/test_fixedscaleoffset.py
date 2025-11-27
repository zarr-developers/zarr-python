from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.numcodecs import ScaleOffset
from zarr.codecs.numcodecs.fixed_scale_offset import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestFixedScaleOffsetCodec(BaseTestCodec):
    test_cls = ScaleOffset
    valid_json_v2 = (
        {"id": "fixedscaleoffset", "scale": 1.0, "offset": 0.0, "astype": "|u1", "dtype": "|u1"},
    )
    valid_json_v3 = (
        {
            "name": "fixedscaleoffset",
            "configuration": {"scale": 1.0, "offset": 0.0, "astype": "uint8", "dtype": "float32"},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
