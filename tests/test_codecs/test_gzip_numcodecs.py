import pytest

from tests.test_codecs.test_gzip import TestGZipCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestGZipNumcodecsCodec(TestGZipCodec):
    test_cls = _numcodecs.GZip  # type: ignore[assignment]
    valid_json_v2 = ({"id": "gzip", "level": 1},)
    valid_json_v3 = (  # type: ignore[assignment]
        {
            "name": "gzip",
            "configuration": {"level": 1},
        },
        {
            "name": "numcodecs.gzip",
            "configuration": {"level": 1},
        },
    )

    @staticmethod
    def check_json_v3(data: object) -> bool:
        from zarr.codecs.numcodecs.gzip import check_json_v3

        return check_json_v3(data)
