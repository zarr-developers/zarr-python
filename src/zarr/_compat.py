import warnings
from collections.abc import Callable
from functools import wraps
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from packaging.version import Version

from zarr.errors import ZarrFutureWarning

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")

# Based off https://github.com/scikit-learn/scikit-learn/blob/e87b32a81c70abed8f2e97483758eb64df8255e9/sklearn/utils/validation.py#L63


def _deprecate_positional_args(
    func: Callable[..., T] | None = None, *, version: str = "3.1.0"
) -> Callable[..., T]:
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default="3.1.0"
        The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f: Callable[..., T]) -> Callable[..., T]:
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args: Any, **kwargs: Any) -> T:
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = [
                f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:], strict=False)
            ]
            formatted_args_msg = ", ".join(args_msg)
            warnings.warn(
                (
                    f"Pass {formatted_args_msg} as keyword args. From version "
                    f"{version} passing these as positional arguments "
                    "will result in an error"
                ),
                ZarrFutureWarning,
                stacklevel=2,
            )
            kwargs.update(zip(sig.parameters, args, strict=False))
            return f(**kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args  # type: ignore[return-value]


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
        return arr.reshape(shape, copy=False)  # type: ignore[call-overload, no-any-return]
    else:
        arr.shape = shape
        return arr
