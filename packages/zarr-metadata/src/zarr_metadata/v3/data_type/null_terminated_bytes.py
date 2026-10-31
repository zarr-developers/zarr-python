"""
zarr-python's `null_terminated_bytes` data type (fixed-length, null-padded bytes).

No specification registers this data type. zarr-python writes it for
NumPy's `S` data types, and warns that its spelling may change
(https://github.com/zarr-developers/zarr-python/blob/a2f2415c86f7d35212ae8c0d5dc5e4a22a6c6191/src/zarr/core/dtype/npy/bytes.py#L385-L390).
This module describes what zarr-python writes. Its definition is in the
`ZARR_PYTHON_CONTEXT` scope, not in `CORE_AND_EXTENSIONS_CONTEXT`.
"""

import base64
from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata._numpy import Base64Bytes
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import DataTypeDefinition, single_byte
from zarr_metadata.v3.data_type._byte import base64_problems

NULL_TERMINATED_BYTES_DATA_TYPE_NAME: Final = "null_terminated_bytes"
"""The `name` field value of the `null_terminated_bytes` data type."""

NullTerminatedBytesDataTypeName = Literal["null_terminated_bytes"]
"""Literal type of the `name` field of the `null_terminated_bytes` data type."""


class NullTerminatedBytesConfiguration(TypedDict, closed=True):
    """
    Configuration for the `null_terminated_bytes` data type.

    Attributes
    ----------
    length_bytes
        The size in bytes of every value, at least 1.
    """

    length_bytes: ReadOnly[int]


class NullTerminatedBytes(TypedDict, closed=True):
    """`null_terminated_bytes` data type metadata."""

    name: NullTerminatedBytesDataTypeName
    configuration: NullTerminatedBytesConfiguration
    must_understand: NotRequired[bool]


NullTerminatedBytesFillValue = Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `null_terminated_bytes`: the bytes, without their trailing null bytes, as a `Base64Bytes` string."""


def _rules(
    configuration: NullTerminatedBytesConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """Check that `length_bytes` is positive. zarr-python refuses a length below 1, since such a type holds no data."""
    length_bytes = configuration["length_bytes"]
    if length_bytes < 1:
        yield ValidationProblem(
            ("length_bytes",), f"expected a positive size, got {length_bytes}", "invalid_value"
        )


def _fill_value_rules(
    configuration: NullTerminatedBytesConfiguration,
    nested: Nested,
    value: NullTerminatedBytesFillValue,
) -> Iterator[ValidationProblem]:
    """Check that the fill value is base64, and that the bytes it encodes fit in `length_bytes`.

    Trailing null bytes are padding, as in `_fill_value_canonical`, so they
    do not count: `"QQAA"` fits in one byte, as `"QQ=="` does.
    """
    problems = list(base64_problems(value))
    if problems:
        yield from problems
        return
    size = len(base64.b64decode(value).rstrip(b"\x00"))
    if size > configuration["length_bytes"]:
        yield ValidationProblem(
            (),
            f"expected at most {configuration['length_bytes']} bytes, got {size}",
            "invalid_value",
        )


def _fill_value_canonical(
    configuration: NullTerminatedBytesConfiguration,
    nested: Nested,
    value: NullTerminatedBytesFillValue,
) -> str:
    """Return the bytes without their trailing null bytes, encoded in base64 again.

    A value is padded with null bytes to `length_bytes`, so `"QQ=="` and
    `"QQAA"` are the same value, as are `"QR=="` and `"QQ=="`.
    """
    return base64.b64encode(base64.b64decode(value).rstrip(b"\x00")).decode("ascii")


NULL_TERMINATED_BYTES_DATA_TYPE: Final = DataTypeDefinition(
    name=NULL_TERMINATED_BYTES_DATA_TYPE_NAME,
    configuration=NullTerminatedBytesConfiguration,
    rules=_rules,
    fill_value=NullTerminatedBytesFillValue,
    fill_value_rules=_fill_value_rules,
    fill_value_canonical=_fill_value_canonical,
    storage=single_byte,
)
"""zarr-python's `null_terminated_bytes` data type: NumPy's `S`. Each value is `length_bytes` single bytes."""


__all__ = [
    "NULL_TERMINATED_BYTES_DATA_TYPE",
    "NULL_TERMINATED_BYTES_DATA_TYPE_NAME",
    "NullTerminatedBytes",
    "NullTerminatedBytesConfiguration",
    "NullTerminatedBytesDataTypeName",
    "NullTerminatedBytesFillValue",
]
