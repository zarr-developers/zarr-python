import pytest

from tests.test_codecs.test_blosc import TestBloscCodec
from zarr.codecs.numcodecs.blosc import Blosc


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrDeprecationWarning")
class TestBloscNumcodecsCodec(TestBloscCodec):
    test_cls = Blosc  # type: ignore[assignment]
    valid_json_v3 = (  # type: ignore[assignment]
        {
            "name": "blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 5,
                "shuffle": "shuffle",
                "typesize": 1,
                "blocksize": 0,
            },
        },
    )
