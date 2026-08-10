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
    """A vectorized (`vindex`) selection used an unsupported form.

    Raised when a `vindex` selection is anything other than a coordinate
    selection (a tuple of integer arrays) or a mask selection (a single
    boolean array) — for example, when it contains a slice. Orthogonal
    (`oindex`) selections are not subject to this restriction.
    """


class BoundsCheckError(IndexError):
    """A selection addressed coordinates outside the domain being indexed.

    Raised for out-of-domain integer indices, slice bounds, index-array
    values, and points passed to `IndexTransform.apply`. Coordinates in this
    algebra are literal: they are never clamped, and a negative value below
    the domain's `inclusive_min` is out of bounds rather than counted from
    the end.
    """
