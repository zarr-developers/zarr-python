"""
zarr-python's `raw_bytes` data type (fixed-length raw bytes).

No specification registers this data type. zarr-python writes it for
NumPy's `V` data types without fields, and warns that its spelling may
change
(https://github.com/zarr-developers/zarr-python/blob/a2f2415c86f7d35212ae8c0d5dc5e4a22a6c6191/src/zarr/core/dtype/npy/bytes.py#L783-L785).
This module describes what zarr-python writes. Its definition is in the
`ZARR_PYTHON_CONTEXT` scope, not in `CORE_AND_EXTENSIONS_CONTEXT`.

Its names start with `ZarrPython` because `RawBytes` names the spec's raw
bits, `r*`, in `zarr_metadata.v3.data_type.raw`.
"""

import base64
from collections.abc import Iterator
from typing import Final

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import DataTypeDefinition, single_byte
from zarr_metadata.v3._types.data_type import (
    ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME,
    ZarrPythonRawBytes,
    ZarrPythonRawBytesConfiguration,
    ZarrPythonRawBytesDataTypeName,
    ZarrPythonRawBytesFillValue,
)
from zarr_metadata.v3.data_type._byte import base64_decoded, base64_problems, base64_spelling


def _rules(
    configuration: ZarrPythonRawBytesConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """Check that `length_bytes` is positive. zarr-python refuses a length below 1, since such a type holds no data."""
    length_bytes = configuration["length_bytes"]
    if length_bytes < 1:
        yield ValidationProblem(
            ("length_bytes",), f"expected a positive size, got {length_bytes}", "invalid_value"
        )


def _fill_value_rules(
    configuration: ZarrPythonRawBytesConfiguration,
    nested: Nested,
    value: ZarrPythonRawBytesFillValue,
) -> Iterator[ValidationProblem]:
    """Check that the fill value is base64, and that the bytes it encodes fit in `length_bytes`.

    zarr-python cuts a longer value to `length_bytes` without a word, which
    loses the bytes past the end, so a longer value is a problem.
    """
    yield from base64_problems(value)
    data = base64_decoded(value)
    if data is not None and len(data) > configuration["length_bytes"]:
        yield ValidationProblem(
            (),
            f"expected at most {configuration['length_bytes']} bytes, got {len(data)}",
            "invalid_value",
        )


def _fill_value_canonical(
    configuration: ZarrPythonRawBytesConfiguration,
    nested: Nested,
    value: ZarrPythonRawBytesFillValue,
) -> str:
    """Return the bytes padded with null bytes to `length_bytes`, encoded in base64 again, as zarr-python writes them.

    zarr-python pads a shorter value with null bytes, so `"QQ=="` and
    `"QQAAAA=="` are the same value in a type of 4 bytes.
    """
    data = base64.b64decode(value).ljust(configuration["length_bytes"], b"\x00")
    return base64_spelling(data)


def _fill_value_default(configuration: ZarrPythonRawBytesConfiguration, nested: Nested) -> str:
    """Return `length_bytes` null bytes in base64, as zarr-python writes the default fill value."""
    return base64_spelling(bytes(configuration["length_bytes"]))


ZARR_PYTHON_RAW_BYTES_DATA_TYPE: Final = DataTypeDefinition(
    name=ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME,
    configuration=ZarrPythonRawBytesConfiguration,
    rules=_rules,
    fill_value=ZarrPythonRawBytesFillValue,
    fill_value_default=_fill_value_default,
    fill_value_rules=_fill_value_rules,
    fill_value_canonical=_fill_value_canonical,
    storage=single_byte,
)
"""zarr-python's `raw_bytes` data type: NumPy's `V` without fields. Each value is `length_bytes` single bytes."""


__all__ = [
    "ZARR_PYTHON_RAW_BYTES_DATA_TYPE",
    "ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME",
    "ZarrPythonRawBytes",
    "ZarrPythonRawBytesConfiguration",
    "ZarrPythonRawBytesDataTypeName",
    "ZarrPythonRawBytesFillValue",
]
