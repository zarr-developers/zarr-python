"""
Zarr v3 `float64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

import re
from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    CANONICAL_NAN_HEX_FLOAT64,
    CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64,
    CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64,
    FLOAT64_DATA_TYPE_NAME,
    Float64DataTypeName,
    Float64FillValue,
    Float64SpecialFillValue,
    HexFloat64,
)
from zarr_metadata.v3.data_type._float import float_fill_value_canonical, float_fill_value_rules

_HEX_FLOAT64_RE: Final = re.compile(r"^0x[0-9a-fA-F]{16}$")


def hex_float64(value: str) -> HexFloat64:
    """Validate `value` as a HexFloat64 and brand it.

    Raises ValueError if `value` is not exactly `0x` followed by 16 hex
    digits.
    """
    if not _HEX_FLOAT64_RE.fullmatch(value):
        raise ValueError(f"Expected '0x' followed by 16 hex digits, got {value!r}")
    return HexFloat64(value)


FLOAT64_DATA_TYPE: Final = DataTypeDefinition(
    name=FLOAT64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Float64FillValue,
    fill_value_rules=float_fill_value_rules("float64", hex_float64),
    fill_value_canonical=float_fill_value_canonical(64),
    storage=multi_byte,
)
"""The `float64` data type. It is written as a bare name and has no configuration. Its fill value is a number, a named value or a hex string."""


__all__ = [
    "CANONICAL_NAN_HEX_FLOAT64",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64",
    "FLOAT64_DATA_TYPE",
    "FLOAT64_DATA_TYPE_NAME",
    "Float64DataTypeName",
    "Float64FillValue",
    "Float64SpecialFillValue",
    "HexFloat64",
    "hex_float64",
]
