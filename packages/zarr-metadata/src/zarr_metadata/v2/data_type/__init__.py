"""The data types zarr-python 2.x writes, one definition per NumPy family."""

from typing import Any, Final

from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type.fixed_width import BYTES_V2, STR_V2, VOID_V2
from zarr_metadata.v2.data_type.object import OBJECT_V2
from zarr_metadata.v2.data_type.scalar import BOOL_V2, COMPLEX_V2, FLOAT_V2, INT_V2, UINT_V2
from zarr_metadata.v2.data_type.struct import STRUCT_V2
from zarr_metadata.v2.data_type.time import DATETIME64_V2, TIMEDELTA64_V2

V2_DATA_TYPES: Final[tuple[ZarrV2DataTypeDefinition[Any], ...]] = (
    BOOL_V2,
    INT_V2,
    UINT_V2,
    FLOAT_V2,
    COMPLEX_V2,
    BYTES_V2,
    STR_V2,
    VOID_V2,
    DATETIME64_V2,
    TIMEDELTA64_V2,
    OBJECT_V2,
    STRUCT_V2,
)
"""Every v2 data type, by family."""

__all__ = [
    "BOOL_V2",
    "BYTES_V2",
    "COMPLEX_V2",
    "DATETIME64_V2",
    "FLOAT_V2",
    "INT_V2",
    "OBJECT_V2",
    "STRUCT_V2",
    "STR_V2",
    "TIMEDELTA64_V2",
    "UINT_V2",
    "V2_DATA_TYPES",
    "VOID_V2",
]
