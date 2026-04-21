"""
Bytes-shaped Zarr v3 data types.

This module covers:

- `bytes` (variable-length bytes, bare string, zarr-extensions)
- `null_terminated_bytes` (fixed-length NUL-terminated, envelope, zarr-extensions)

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
"""

from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

BYTES_DTYPE_NAME: Final = "bytes"
"""The `data_type` value for the variable-length `bytes` type."""

BytesDTypeName = Literal["bytes"]
"""Literal type of the `data_type` field for `bytes`."""

NULL_TERMINATED_BYTES_DTYPE_NAME: Final = "null_terminated_bytes"
"""The `name` field value of the `null_terminated_bytes` data type."""

NullTerminatedBytesDTypeName = Literal["null_terminated_bytes"]
"""Literal type of the `name` field of the `null_terminated_bytes` data type."""


class FixedLengthBytesConfig(TypedDict):
    """
    Configuration for fixed-length bytes-shaped data types.

    Used by `null_terminated_bytes` (and any other fixed-length bytes
    extension). `length_bytes` is the allocated byte count per element.
    """

    length_bytes: ReadOnly[int]


class NullTerminatedBytes(TypedDict):
    """`null_terminated_bytes` data type metadata."""

    name: NullTerminatedBytesDTypeName
    configuration: FixedLengthBytesConfig


__all__ = [
    "BYTES_DTYPE_NAME",
    "NULL_TERMINATED_BYTES_DTYPE_NAME",
    "BytesDTypeName",
    "FixedLengthBytesConfig",
    "NullTerminatedBytes",
    "NullTerminatedBytesDTypeName",
]
