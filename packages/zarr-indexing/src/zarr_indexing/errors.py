"""Canonical index-error types raised by the transform algebra.

Both subclass the built-in `IndexError`, so an `except IndexError` catch site
keeps working unchanged whichever library raised.

`zarr.errors` defines classes of the same names, and they are *not* these
objects: `zarr.errors.BoundsCheckError is BoundsCheckError` is false. Catching
zarr's around a call into this package therefore catches nothing but their
shared `IndexError` base. Import these from here.
"""

from __future__ import annotations

__all__ = [
    "BoundsCheckError",
    "VindexInvalidSelectionError",
]


class VindexInvalidSelectionError(IndexError):
    """A wrapper `vindex` selection contained a slice.

    Raised by `LazyArray`'s selection validation: the wrapper's vectorized
    dialect accepts coordinate selections (integer arrays, with scalars and
    an ellipsis) or a single boolean mask, and rejects slices with this
    error. Other invalid entries raise plain `IndexError`, and the
    engine-level `IndexTransform.vindex` is wider — it accepts residual
    slice dimensions without raising.

    Examples
    --------
    Raised by the wrapper, not the engine — a slice inside `vindex`:

    >>> import numpy as np
    >>> from zarr_indexing import LazyArray
    >>> view = LazyArray.from_numpy(np.arange(12).reshape(3, 4))
    >>> view.lazy.vindex[np.array([0, 2]), :]
    Traceback (most recent call last):
        ...
    zarr_indexing.errors.VindexInvalidSelectionError: ...
    """


class BoundsCheckError(IndexError):
    """A selection addressed coordinates outside the domain being indexed.

    Raised for out-of-domain integer indices, slice bounds, index-array
    values, and points passed to `IndexTransform.apply`. Coordinates in this
    algebra are literal: they are never clamped, and a negative value below
    the domain's `inclusive_min` is out of bounds rather than counted from
    the end.

    Examples
    --------
    >>> from zarr_indexing import IndexTransform
    >>> IndexTransform.from_shape((8,)).apply((9,))
    Traceback (most recent call last):
        ...
    zarr_indexing.errors.BoundsCheckError: ...

    A negative index is a literal coordinate, not "from the end":

    >>> IndexTransform.from_shape((8,))[-1]
    Traceback (most recent call last):
        ...
    zarr_indexing.errors.BoundsCheckError: ...
    """
