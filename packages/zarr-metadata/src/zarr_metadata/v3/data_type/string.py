"""
Zarr `string` data type (variable-length utf-8, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    variable_length,
)
from zarr_metadata.v3._types.data_type import (
    STRING_DATA_TYPE_NAME,
    StringDataTypeName,
    StringFillValue,
)

STRING_DATA_TYPE: Final = DataTypeDefinition(
    name=STRING_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=StringFillValue,
    storage=variable_length,
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
