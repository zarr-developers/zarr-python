from typing import Any
from warnings import warn

import numpy as np

try:
    STRING_DTYPE = np.dtype("T")
    NUMPY_SUPPORTS_VLEN_STRING = True
except TypeError:
    STRING_DTYPE = np.dtype("object")
    NUMPY_SUPPORTS_VLEN_STRING = False


def cast_to_string_dtype(
    data: np.ndarray[Any, np.dtype[Any]], safe: bool = False
) -> np.ndarray[Any, np.dtype[Any]]:
    if np.issubdtype(data.dtype, np.str_):
        return data
    if np.issubdtype(data.dtype, np.object_):
        if NUMPY_SUPPORTS_VLEN_STRING:
            try:
                # cast to variable-length string dtype, fail if object contains non-string data
                # mypy says "error: Unexpected keyword argument "coerce" for "StringDType"  [call-arg]"
                return data.astype(np.dtypes.StringDType(coerce=False), copy=False)  # type: ignore[call-arg]
            except ValueError as e:
                raise ValueError("Cannot cast object dtype to string dtype") from e
        else:
            out = data.astype(np.str_)
            if not safe:
                warn(
                    f"Casted object dtype to string dtype {out.dtype}. To avoid this warning, "
                    "cast the data to a string dtype before passing to Zarr or upgrade to NumPy >= 2.",
                    stacklevel=2,
                )
            return out
    raise ValueError(f"Cannot cast dtype {data.dtype} to string dtype")
