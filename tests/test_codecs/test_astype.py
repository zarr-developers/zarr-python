import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.numcodecs.astype import AsType, check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestAsType(BaseTestCodec):
    test_cls = AsType
    valid_json_v2 = ({"id": "astype", "encode_dtype": "<f4", "decode_dtype": "<f4"},)
    valid_json_v3 = (
        {
            "name": "astype",
            "configuration": {"encode_dtype": "float32", "decode_dtype": "float64"},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)


def test_v3_json_alias() -> None:
    """
    Test that the default JSON output of the legacy numcodecs.zarr3.AsType codec is readable, even if it's
    underspecified.
    """
    assert AsType.from_json(
        {
            "name": "numcodecs.astype",
            "configuration": {"encode_dtype": ">i2", "decode_dtype": "|i1"},
        }
    ) == AsType(encode_dtype=">i2", decode_dtype="|i1")
