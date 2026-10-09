"""
Zarr `bytes` data type (variable-length raw bytes, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md
"""

import base64
import re
from collections.abc import Iterator
from typing import Final, Literal, NewType

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    EmptyConfiguration,
    Nested,
    variable_length,
)
from zarr_metadata.v3.data_type._byte import ByteValue

BYTES_DATA_TYPE_NAME: Final = "bytes"
"""The `data_type` value for the variable-length `bytes` type."""

BytesDataTypeName = Literal["bytes"]
"""Literal type of the `data_type` field for `bytes`."""

Base64Bytes = NewType("Base64Bytes", str)
"""A standard-alphabet base64-encoded byte sequence."""

_BASE64_RE: Final = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def base64_bytes(value: str) -> Base64Bytes:
    """Validate `value` as a Base64Bytes and brand it.

    Raises ValueError if `value` is not standard-alphabet base64
    (length must be a multiple of 4 once padded; only `A-Z`, `a-z`,
    `0-9`, `+`, `/`, and trailing `=` padding are permitted).
    """
    if len(value) % 4 != 0 or not _BASE64_RE.fullmatch(value):
        raise ValueError(f"Expected standard-alphabet base64, got {value!r}")
    return Base64Bytes(value)


BytesFillValue = tuple[ByteValue, ...] | Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `bytes`.

Either a JSON array of integers in `[0, 255]` (one per byte), or a
`Base64Bytes` string encoding the byte sequence.
"""


def _fill_value_rules(
    configuration: EmptyConfiguration, nested: Nested, value: BytesFillValue
) -> Iterator[ValidationProblem]:
    """A string of standard-alphabet base64, when it is not byte values."""
    if not isinstance(value, str):
        return
    try:
        base64_bytes(value)
    except ValueError:
        yield ValidationProblem(
            (), f"expected standard-alphabet base64, got {shown(value)}", "invalid_value"
        )


def _fill_value_canonical(
    configuration: EmptyConfiguration, nested: Nested, value: BytesFillValue
) -> str:
    """The bytes, however written, as the base64 that encodes them.

    The string "is more compact"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md?plain=1#L11),
    and encoding the bytes again spells them one way: `"QR=="` decodes to
    the one byte `"QQ=="` does.
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
"""The `bytes` data type: a bare name, with nothing to configure.

Its values are "variable-length byte strings"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md?plain=1#L3),
each stored in as many bytes as it needs.
"""


__all__ = [
    "BYTES_DATA_TYPE",
    "BYTES_DATA_TYPE_NAME",
    "Base64Bytes",
    "BytesDataTypeName",
    "BytesFillValue",
    "base64_bytes",
]
