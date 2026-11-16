"""
Zarr `bytes` data type (variable-length raw bytes, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md
"""

import base64
from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata._numpy import Base64Bytes, base64_bytes
from zarr_metadata.definition._definition import (
    ConstantFillValue,
    Nested,
)
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    variable_length,
)
from zarr_metadata.v3._types.data_type import (
    BYTES_DATA_TYPE_NAME,
    BytesDataTypeName,
    BytesFillValue,
)
from zarr_metadata.v3.data_type._byte import (
    base64_fill_value_rules,
    base64_spelling,
    vlen_bytes_codec,
)


def _fill_value_canonical(
    configuration: EmptyConfiguration, nested: Nested, value: BytesFillValue
) -> str:
    """Return the bytes as the base64 string that encodes them, however they were written.

    The string "is more compact"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md?plain=1#L11),
    and encoding the bytes again gives a single spelling: `"QR=="` decodes
    to the same byte as `"QQ=="`.
    """
    data = base64.b64decode(value) if isinstance(value, str) else bytes(value)
    return base64_spelling(data)


BYTES_DATA_TYPE: Final = DataTypeDefinition(
    name=BYTES_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=BytesFillValue,
    fill_value_default=ConstantFillValue(""),
    fill_value_rules=base64_fill_value_rules,
    fill_value_canonical=_fill_value_canonical,
    storage=variable_length,
    array_bytes_codec_default=vlen_bytes_codec,
    number_class="non_numeric",
)
"""The `bytes` data type. It is written as a bare name and has no configuration.

Its values are "variable-length byte strings"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md?plain=1#L3),
and each value is stored in as many bytes as it needs.
"""


__all__ = [
    "BYTES_DATA_TYPE",
    "BYTES_DATA_TYPE_NAME",
    "Base64Bytes",
    "BytesDataTypeName",
    "BytesFillValue",
    "base64_bytes",
    "base64_fill_value_rules",
]
