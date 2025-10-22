from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.numcodecs.shuffle import Shuffle, check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestShuffle(BaseTestCodec):
    test_cls = Shuffle
    valid_json_v2 = ({"id": "shuffle", "elementsize": 4},)
    valid_json_v3 = (
        {
            "name": "shuffle",
            "configuration": {"elementsize": 4},
        },
        {
            "name": "numcodecs.shuffle",
            "configuration": {"elementsize": 4},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
