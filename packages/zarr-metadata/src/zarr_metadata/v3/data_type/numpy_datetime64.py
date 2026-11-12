"""
Zarr `numpy.datetime64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md
"""

from typing import Final

from zarr_metadata._numpy import (
    numpy_time_fill_value_canonical,
)
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import DataTypeDefinition
from zarr_metadata.v3._types.data_type import (
    NUMPY_DATETIME64_DATA_TYPE_NAME,
    NUMPY_TIME_UNIT,
    NumpyDatetime64,
    NumpyDatetime64Configuration,
    NumpyDatetime64DataTypeName,
    NumpyDatetime64FillValue,
    NumpyTimeUnit,
)
from zarr_metadata.v3.data_type._numpy_time import numpy_time_storage

NUMPY_DATETIME64_DATA_TYPE: Final = DataTypeDefinition(
    name=NUMPY_DATETIME64_DATA_TYPE_NAME,
    configuration=NumpyDatetime64Configuration,
    fill_value=NumpyDatetime64FillValue,
    fill_value_default=ConstantFillValue("NaT"),
    fill_value_canonical=numpy_time_fill_value_canonical,
    storage=numpy_time_storage,
    number_class="integral",
)
"""The `numpy.datetime64` data type.

Its number class is integral, since it "is compatible with any codec that
supports arrays of signed 64-bit integers"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L120).
"""


__all__ = [
    "NUMPY_DATETIME64_DATA_TYPE",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NUMPY_TIME_UNIT",
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
]
