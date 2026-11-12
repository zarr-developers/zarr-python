"""
Zarr `string` data type (variable-length utf-8, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    variable_length,
)
from zarr_metadata.v3._types.codec import VLEN_UTF8_CODEC_NAME, VlenUtf8CodecObject
from zarr_metadata.v3._types.data_type import (
    STRING_DATA_TYPE_NAME,
    StringDataTypeName,
    StringFillValue,
)


def _array_bytes_codec_default(*_: object) -> VlenUtf8CodecObject:
    """Return the `vlen-utf8` codec, as zarr-python chooses it (https://github.com/zarr-developers/zarr-python/blob/0981311c07f0a8dc8650b74e33bd385ccb411f26/src/zarr/core/array.py#L5114-L5139)."""
    return {"name": VLEN_UTF8_CODEC_NAME}


STRING_DATA_TYPE: Final = DataTypeDefinition(
    name=STRING_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=StringFillValue,
    fill_value_default=ConstantFillValue(""),
    storage=variable_length,
    array_bytes_codec_default=_array_bytes_codec_default,
)
"""The `string` data type. It is written as a bare name and has no configuration.

Its values are "variable-length UTF8 strings"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md?plain=1#L3),
and each value is stored in as many bytes as it needs.
"""


__all__ = [
    "STRING_DATA_TYPE",
    "STRING_DATA_TYPE_NAME",
    "StringDataTypeName",
    "StringFillValue",
]
