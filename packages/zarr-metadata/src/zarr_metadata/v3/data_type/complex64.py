"""
Zarr v3 `complex64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    COMPLEX64_DATA_TYPE_NAME,
    Complex64Component,
    Complex64DataTypeName,
    Complex64FillValue,
)
from zarr_metadata.v3.data_type._float import (
    complex_fill_value_canonical,
    complex_fill_value_rules,
)
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE

COMPLEX64_DATA_TYPE: Final = DataTypeDefinition(
    name=COMPLEX64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Complex64FillValue,
    fill_value_default=ConstantFillValue((0.0, 0.0)),
    fill_value_rules=complex_fill_value_rules(FLOAT32_DATA_TYPE.fill_value_rules),
    fill_value_canonical=complex_fill_value_canonical(FLOAT32_DATA_TYPE.fill_value_canonical),
    storage=multi_byte,
)
"""The `complex64` data type. It is written as a bare name and has no configuration. Its fill value is a pair of `float32` components."""


__all__ = [
    "COMPLEX64_DATA_TYPE",
    "COMPLEX64_DATA_TYPE_NAME",
    "Complex64Component",
    "Complex64DataTypeName",
    "Complex64FillValue",
]
