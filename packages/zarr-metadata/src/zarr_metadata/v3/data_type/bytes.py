"""
Zarr `bytes` data type (variable-length raw bytes, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md
"""

import base64
from collections.abc import Iterator
from typing import Final, Literal

from zarr_metadata._json import ValidationProblem
from zarr_metadata._numpy import Base64Bytes, base64_bytes
from zarr_metadata.definition._definition import (
    EmptyConfiguration,
    Nested,
)
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    variable_length,
)
from zarr_metadata.v3.data_type._byte import ByteValue, base64_problems

BYTES_DATA_TYPE_NAME: Final = "bytes"
"""The `data_type` value for the variable-length `bytes` type."""

BytesDataTypeName = Literal["bytes"]
"""Literal type of the `data_type` field for `bytes`."""

BytesFillValue = tuple[ByteValue, ...] | Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `bytes`.

Either a JSON array of integers in `[0, 255]` (one per byte), or a
`Base64Bytes` string encoding the byte sequence.
"""


def _fill_value_rules(
    configuration: EmptyConfiguration, nested: Nested, value: BytesFillValue
) -> Iterator[ValidationProblem]:
    """Check that a string fill value is standard-alphabet base64. A fill value written as byte values is not checked here."""
    if isinstance(value, str):
        yield from base64_problems(value)


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
    return base64.b64encode(data).decode("ascii")


BYTES_DATA_TYPE: Final = DataTypeDefinition(
    name=BYTES_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=BytesFillValue,
    fill_value_rules=_fill_value_rules,
    fill_value_canonical=_fill_value_canonical,
    storage=variable_length,
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
]
