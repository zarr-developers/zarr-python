from __future__ import annotations

import pytest

from tests.test_codecs.test_zstd import TestZstdCodec
from zarr.codecs.numcodecs import Zstd


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrDeprecationWarning")
class TestNumcodecsZstdCodec(TestZstdCodec):
    test_cls = Zstd  # type: ignore[assignment]
    valid_json_v3 = (  # type: ignore[assignment]
        {
            "name": "zstd",
            "configuration": {
                "level": 0,
                "checksum": False,
            },
        },
        {
            "name": "numcodecs.zstd",
            "configuration": {
                "level": 0,
                "checksum": False,
            },
        },
    )

    @staticmethod
    def check_json_v3(data: object) -> bool:
        from zarr.codecs.numcodecs.zstd import check_json_v3

        return check_json_v3(data)
