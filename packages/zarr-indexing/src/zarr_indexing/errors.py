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


class VindexInvalidSelectionError(IndexError): ...


class BoundsCheckError(IndexError): ...
