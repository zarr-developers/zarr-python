import warnings

from zarr.api.synchronous import (
    array,
    create,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    open_array,
    open_like,
    zeros,
    zeros_like,
)

__all__ = [
    "array",
    "create",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "open_array",
    "open_like",
    "zeros",
    "zeros_like",
]

warnings.warn(
    "zarr.creation is deprecated, use zarr.api.synchronous",
    DeprecationWarning,
    stacklevel=2,
)
