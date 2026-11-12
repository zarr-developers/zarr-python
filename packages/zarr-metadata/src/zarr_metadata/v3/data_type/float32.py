"""
Zarr v3 `float32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

import re
from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    CANONICAL_NAN_HEX_FLOAT32,
    CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32,
    CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32,
    FLOAT32_DATA_TYPE_NAME,
    Float32DataTypeName,
    Float32FillValue,
    Float32SpecialFillValue,
    HexFloat32,
)
from zarr_metadata.v3.data_type._float import float_fill_value_canonical, float_fill_value_rules

_HEX_FLOAT32_RE: Final = re.compile(r"^0x[0-9a-fA-F]{8}$")


def hex_float32(value: str) -> HexFloat32:
    """Validate `value` as a HexFloat32 and brand it.

    Raises ValueError if `value` is not exactly `0x` followed by 8 hex
    digits.
    """
    if not _HEX_FLOAT32_RE.fullmatch(value):
        raise ValueError(f"Expected '0x' followed by 8 hex digits, got {value!r}")
    return HexFloat32(value)


FLOAT32_DATA_TYPE: Final = DataTypeDefinition(
    name=FLOAT32_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Float32FillValue,
    fill_value_default=ConstantFillValue(0.0),
    fill_value_rules=float_fill_value_rules("float32", hex_float32),
    fill_value_canonical=float_fill_value_canonical(32),
    storage=multi_byte,
)
"""The `float32` data type. It is written as a bare name and has no configuration. Its fill value is a number, a named value or a hex string."""


__all__ = [
    "CANONICAL_NAN_HEX_FLOAT32",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32",
    "FLOAT32_DATA_TYPE",
    "FLOAT32_DATA_TYPE_NAME",
    "Float32DataTypeName",
    "Float32FillValue",
    "Float32SpecialFillValue",
    "HexFloat32",
    "hex_float32",
]
