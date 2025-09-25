from typing import TypeGuard

import pytest

from tests.test_codecs.test_blosc import TestBloscCodec
from zarr.codecs.numcodecs.blosc import Blosc, BloscConfigV3_Legacy, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestBloscNumcodecsCodec(TestBloscCodec):
    test_cls = Blosc
    valid_json_v3 = (
        {
            "name": "blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 5,
                "shuffle": 1,
                "blocksize": 0,
            },
        },
        {
            "name": "numcodecs.blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 5,
                "shuffle": 1,
                "blocksize": 0,
            },
        },
    )

    @staticmethod
    def check_json_v3(data: object) -> TypeGuard[BloscConfigV3_Legacy]:
        return check_json_v3(data)
