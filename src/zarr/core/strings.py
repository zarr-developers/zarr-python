"""This module contains utilities for working with string arrays across
different versions of Numpy.
"""

from typing import Any, Union, cast
from warnings import warn

import numpy as np

# _STRING_DTYPE is the in-memory datatype that will be used for V3 string arrays
# when reading data back from Zarr.
# Any valid string-like datatype should be fine for *setting* data.

_STRING_DTYPE: Union["np.dtypes.StringDType", "np.dtypes.ObjectDType"]
_NUMPY_SUPPORTS_VLEN_STRING: bool


def cast_array(
    data: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, Union["np.dtypes.StringDType", "np.dtypes.ObjectDType"]]:
    raise NotImplementedError


try:
    # this new vlen string dtype was added in NumPy 2.0
    _STRING_DTYPE = np.dtypes.StringDType()
    _NUMPY_SUPPORTS_VLEN_STRING = True

    def cast_array(
        data: np.ndarray[Any, np.dtype[Any]],
    ) -> np.ndarray[Any, np.dtypes.StringDType | np.dtypes.ObjectDType]:
        out = data.astype(_STRING_DTYPE, copy=False)
        return cast(np.ndarray[Any, np.dtypes.StringDType], out)

except AttributeError:
    # if not available, we fall back on an object array of strings, as in Zarr < 3
    _STRING_DTYPE = np.dtypes.ObjectDType()
    _NUMPY_SUPPORTS_VLEN_STRING = False

    def cast_array(
        data: np.ndarray[Any, np.dtype[Any]],
    ) -> np.ndarray[Any, Union["np.dtypes.StringDType", "np.dtypes.ObjectDType"]]:
        out = data.astype(_STRING_DTYPE, copy=False)
        return cast(np.ndarray[Any, np.dtypes.ObjectDType], out)


def cast_to_string_dtype(
    data: np.ndarray[Any, np.dtype[Any]], safe: bool = False
) -> np.ndarray[Any, Union["np.dtypes.StringDType", "np.dtypes.ObjectDType"]]:
    """Take any data and attempt to cast to to our preferred string dtype.

    data :  np.ndarray
        The data to cast

    safe : bool
        If True, do not issue a warning if the data is cast from object to string dtype.

    """
    if np.issubdtype(data.dtype, np.str_):
        # legacy fixed-width string type (e.g. "<U10")
        return cast_array(data)
        # out = data.astype(STRING_DTYPE, copy=False)
        # return cast(np.ndarray[Any, np.dtypes.StringDType | np.dtypes.ObjectDType], out)
    if _NUMPY_SUPPORTS_VLEN_STRING and np.issubdtype(data.dtype, _STRING_DTYPE):
        # already a valid string variable length string dtype
        return cast_array(data)
    if np.issubdtype(data.dtype, np.object_):
        # object arrays require more careful handling
        if _NUMPY_SUPPORTS_VLEN_STRING:
            try:
                # cast to variable-length string dtype, fail if object contains non-string data
                # mypy says "error: Unexpected keyword argument "coerce" for "StringDType"  [call-arg]"
                # also: Value of type variable "_ScalarType" of "astype" of "ndarray" cannot be "str"  [type-var]
                out = data.astype(np.dtypes.StringDType(coerce=False), copy=False)  # type: ignore[call-arg,type-var]
                return cast_array(out)
            except ValueError as e:
                raise ValueError("Cannot cast object dtype to string dtype") from e
        else:
            if not safe:
                warn(
                    "Treating object array as valid string array. To avoid this warning, "
                    "cast the data to a string dtype before passing to Zarr or upgrade to NumPy >= 2.",
                    stacklevel=2,
                )
            return cast_array(data)
    raise ValueError(f"Cannot cast dtype {data.dtype} to string dtype")
