"""
zarr-python's `variable_length_bytes` data type (variable-length raw bytes).

No specification registers this name. zarr-python writes it for arrays of
Python `bytes` objects, and warns that its spelling may change
(https://github.com/zarr-developers/zarr-python/blob/a2f2415c86f7d35212ae8c0d5dc5e4a22a6c6191/src/zarr/core/dtype/npy/bytes.py#L1141-L1143).
zarr-extensions registers the same values as `bytes`. This module describes
what zarr-python writes. Its definition is in the `ZARR_PYTHON_CONTEXT` scope, not
in `CORE_AND_EXTENSIONS_CONTEXT`.
"""

import base64
from collections.abc import Iterator
from typing import Final, Literal

from zarr_metadata._json import ValidationProblem
from zarr_metadata._numpy import Base64Bytes
from zarr_metadata.definition._definition import EmptyConfiguration, Nested
from zarr_metadata.v3._definition import DataTypeDefinition, variable_length
from zarr_metadata.v3.data_type._byte import base64_problems

VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME: Final = "variable_length_bytes"
"""The `data_type` value for the `variable_length_bytes` type."""

VariableLengthBytesDataTypeName = Literal["variable_length_bytes"]
"""Literal type of the `data_type` field for `variable_length_bytes`."""

VariableLengthBytesFillValue = Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `variable_length_bytes`: a `Base64Bytes` string, which is what zarr-python writes and reads."""


def _fill_value_rules(
    configuration: EmptyConfiguration, nested: Nested, value: VariableLengthBytesFillValue
) -> Iterator[ValidationProblem]:
    """Check that the fill value is standard-alphabet base64."""
    yield from base64_problems(value)


def _fill_value_canonical(
    configuration: EmptyConfiguration, nested: Nested, value: VariableLengthBytesFillValue
) -> str:
    """Return the bytes encoded in base64 again, so `"QR=="` becomes `"QQ=="`."""
    return base64.b64encode(base64.b64decode(value)).decode("ascii")


VARIABLE_LENGTH_BYTES_DATA_TYPE: Final = DataTypeDefinition(
    name=VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=VariableLengthBytesFillValue,
    fill_value_rules=_fill_value_rules,
    fill_value_canonical=_fill_value_canonical,
    storage=variable_length,
)
"""zarr-python's `variable_length_bytes` data type. It is written as a bare name and has no configuration, and each value is stored in as many bytes as it needs."""


__all__ = [
    "VARIABLE_LENGTH_BYTES_DATA_TYPE",
    "VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME",
    "VariableLengthBytesDataTypeName",
    "VariableLengthBytesFillValue",
]
