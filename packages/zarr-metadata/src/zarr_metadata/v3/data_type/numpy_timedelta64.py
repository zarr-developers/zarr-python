"""
Zarr `numpy.timedelta64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.timedelta64/README.md
"""

from typing import Final

from zarr_metadata._numpy import (
    numpy_time_fill_value_canonical,
)
from zarr_metadata.v3._definition import DataTypeDefinition
from zarr_metadata.v3._types.data_type import (
    NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    NumpyTimedelta64,
    NumpyTimedelta64Configuration,
    NumpyTimedelta64DataTypeName,
    NumpyTimedelta64FillValue,
)
from zarr_metadata.v3.data_type._numpy_time import numpy_time_storage

NUMPY_TIMEDELTA64_DATA_TYPE: Final = DataTypeDefinition(
    name=NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    configuration=NumpyTimedelta64Configuration,
    fill_value=NumpyTimedelta64FillValue,
    fill_value_canonical=numpy_time_fill_value_canonical,
    storage=numpy_time_storage,
)
"""The `numpy.timedelta64` data type."""


__all__ = [
    "NUMPY_TIMEDELTA64_DATA_TYPE",
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NumpyTimedelta64",
    "NumpyTimedelta64Configuration",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
]
