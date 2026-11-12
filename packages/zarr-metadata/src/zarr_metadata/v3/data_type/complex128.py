"""
Zarr v3 `complex128` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    COMPLEX128_DATA_TYPE_NAME,
    Complex128Component,
    Complex128DataTypeName,
    Complex128FillValue,
)
from zarr_metadata.v3.data_type._float import (
    complex_fill_value_canonical,
    complex_fill_value_rules,
)
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE

COMPLEX128_DATA_TYPE: Final = DataTypeDefinition(
    name=COMPLEX128_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Complex128FillValue,
    fill_value_rules=complex_fill_value_rules(FLOAT64_DATA_TYPE.fill_value_rules),
    fill_value_canonical=complex_fill_value_canonical(FLOAT64_DATA_TYPE.fill_value_canonical),
    storage=multi_byte,
)
"""The `complex128` data type. It is written as a bare name and has no configuration. Its fill value is a pair of `float64` components."""


__all__ = [
    "COMPLEX128_DATA_TYPE",
    "COMPLEX128_DATA_TYPE_NAME",
    "Complex128Component",
    "Complex128DataTypeName",
    "Complex128FillValue",
]
