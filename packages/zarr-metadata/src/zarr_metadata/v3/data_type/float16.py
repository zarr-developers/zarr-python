"""
Zarr v3 `float16` data type.

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
    CANONICAL_NAN_HEX_FLOAT16,
    CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT16,
    CANONICAL_POSITIVE_INFINITY_HEX_FLOAT16,
    FLOAT16_DATA_TYPE_NAME,
    Float16DataTypeName,
    Float16FillValue,
    Float16SpecialFillValue,
    HexFloat16,
)
from zarr_metadata.v3.data_type._float import float_fill_value_canonical, float_fill_value_rules

_HEX_FLOAT16_RE: Final = re.compile(r"^0x[0-9a-fA-F]{4}$")


def hex_float16(value: str) -> HexFloat16:
    """Validate `value` as a HexFloat16 and brand it.

    Raises ValueError if `value` is not exactly `0x` followed by 4 hex
    digits.
    """
    if not _HEX_FLOAT16_RE.fullmatch(value):
        raise ValueError(f"Expected '0x' followed by 4 hex digits, got {value!r}")
    return HexFloat16(value)


FLOAT16_DATA_TYPE: Final = DataTypeDefinition(
    name=FLOAT16_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Float16FillValue,
    fill_value_default=ConstantFillValue(0.0),
    fill_value_rules=float_fill_value_rules("float16", hex_float16),
    fill_value_canonical=float_fill_value_canonical(16),
    storage=multi_byte,
)
"""The `float16` data type. It is written as a bare name and has no configuration. Its fill value is a number, a named value or a hex string."""


__all__ = [
    "CANONICAL_NAN_HEX_FLOAT16",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT16",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT16",
    "FLOAT16_DATA_TYPE",
    "FLOAT16_DATA_TYPE_NAME",
    "Float16DataTypeName",
    "Float16FillValue",
    "Float16SpecialFillValue",
    "HexFloat16",
    "hex_float16",
]
