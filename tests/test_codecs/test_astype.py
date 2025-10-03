from collections.abc import Mapping
from typing import TypeGuard

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs.numcodecs.astype import AsType, AsTypeJSON_V3, check_json_v2
from zarr.core.common import _check_codecjson_v3
from zarr.core.dtype.common import check_dtype_spec_v3


def check_json_v3(data: object) -> TypeGuard[AsTypeJSON_V3]:
    """
    A type guard for the Zarr V3 form of the Astype codec JSON.

    This check is more strict than the one we use for input data, because we are using it
    to check output data.
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] == "numcodecs.astype"
        and "configuration" in data
        and "encode_dtype" in data["configuration"]
        and "decode_dtype" in data["configuration"]
        and check_dtype_spec_v3(data["configuration"]["decode_dtype"])
        and check_dtype_spec_v3(data["configuration"]["encode_dtype"])
    )


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestAsType(BaseTestCodec):
    test_cls = AsType
    valid_json_v2 = ({"id": "astype", "encode_dtype": "<f4", "decode_dtype": "<f4"},)
    valid_json_v3 = (
        {
            "name": "astype",
            "configuration": {"encode_dtype": "float32", "decode_dtype": "float64"},
        },
        {
            "name": "numcodecs.astype",
            "configuration": {"encode_dtype": "|u1", "decode_dtype": "|u1"},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)
