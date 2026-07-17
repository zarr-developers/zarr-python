"""Canonical index-error types raised by the transform algebra.

These are the authoritative class definitions. `zarr.errors` re-exports the
same objects (`from zarr_transforms.errors import ...`) so that, e.g.,
`zarr.errors.BoundsCheckError is zarr_transforms.errors.BoundsCheckError`.
Both subclass the built-in `IndexError`, so existing `except IndexError` (or
`except zarr.errors.BoundsCheckError`) catch sites keep working unchanged.
"""

from __future__ import annotations

__all__ = [
    "BoundsCheckError",
    "VindexInvalidSelectionError",
]


class VindexInvalidSelectionError(IndexError): ...


class BoundsCheckError(IndexError): ...
