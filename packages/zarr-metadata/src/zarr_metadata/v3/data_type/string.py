"""
Zarr `string` data type (variable-length utf-8, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, variable_length

STRING_DATA_TYPE_NAME: Final = "string"
"""The `data_type` value for the `string` type."""

StringDataTypeName = Literal["string"]
"""Literal type of the `data_type` field for `string`."""

StringFillValue = str
"""Permitted JSON shape of the `fill_value` field for `string`: a JSON unicode string."""


STRING_DATA_TYPE: Final = DataTypeDefinition(
    name=STRING_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=StringFillValue,
    storage=variable_length,
)
"""The `string` data type: a bare name, with nothing to configure.

Its values are "variable-length UTF8 strings"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md?plain=1#L3),
each stored in as many bytes as it needs.
"""


__all__ = [
    "STRING_DATA_TYPE",
    "STRING_DATA_TYPE_NAME",
    "StringDataTypeName",
    "StringFillValue",
]
