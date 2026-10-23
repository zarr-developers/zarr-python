"""The data types that zarr-python 2.x writes, with one definition per NumPy family.

Each family's definition is in its submodule (`fixed_width`, `object`,
`scalar`, `struct`, `time`). This module holds the dtype's JSON shape,
`ZarrV2DataTypeMetadata`, and `DATA_TYPES_V2`, the tuple of every
definition.
"""

from typing import Any, Final

from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2._field_json import ZarrV2DataTypeMetadata
from zarr_metadata.v2.data_type.fixed_width import BYTES_V2, STR_V2, VOID_V2
from zarr_metadata.v2.data_type.object import OBJECT_V2
from zarr_metadata.v2.data_type.scalar import BOOL_V2, COMPLEX_V2, FLOAT_V2, INT_V2, UINT_V2
from zarr_metadata.v2.data_type.struct import STRUCT_V2
from zarr_metadata.v2.data_type.time import DATETIME64_V2, TIMEDELTA64_V2

DATA_TYPES_V2: Final[tuple[ZarrV2DataTypeDefinition[Any], ...]] = (
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
    "DATA_TYPES_V2",
    "ZarrV2DataTypeMetadata",
]
