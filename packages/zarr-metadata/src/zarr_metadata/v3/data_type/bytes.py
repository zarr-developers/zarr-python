"""
Zarr `bytes` data type (variable-length raw bytes, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/bytes
"""

import re
from typing import Final, Literal, NewType

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


BytesFillValue = tuple[int, ...] | Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `bytes`.

Either a JSON array of integers in `[0, 255]` (one per byte), or a
`Base64Bytes` string encoding the byte sequence.
"""


__all__ = [
    "BYTES_DATA_TYPE_NAME",
    "Base64Bytes",
    "BytesDataTypeName",
    "BytesFillValue",
    "base64_bytes",
]
