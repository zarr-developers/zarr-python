from __future__ import annotations

from tests.test_codecs.test_zstd import TestZstdCodec
from zarr.codecs.numcodecs import Zstd


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
