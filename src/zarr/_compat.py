from typing import TYPE_CHECKING, Any

import numpy as np
from packaging.version import Version

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _reshape_view(arr: "NDArray[Any]", shape: tuple[int, ...]) -> "NDArray[Any]":
    """Reshape an array without copying data.

    This function provides compatibility across NumPy versions for reshaping arrays
    as views. On NumPy >= 2.1, it uses ``reshape(copy=False)`` which explicitly
    fails if a view cannot be created. On older versions, it uses direct shape
    assignment which has the same behavior but is deprecated in 2.5+.

    Parameters
    ----------
    arr : NDArray
        The array to reshape.
    shape : tuple of int
        The new shape.

    Returns
    -------
    NDArray
        A reshaped view of the array.

    Raises
    ------
    AttributeError
        If a view cannot be created (the array is not contiguous) on NumPy < 2.1.
    ValueError
        If a view cannot be created (the array is not contiguous) on NumPy >= 2.1.
    """
    if Version(np.__version__) >= Version("2.1"):
        return arr.reshape(shape, copy=False)
    else:
        arr.shape = shape
        return arr
