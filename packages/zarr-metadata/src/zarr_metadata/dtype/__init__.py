"""
Zarr data type spec types.
"""

from collections.abc import Mapping, Sequence

# Wider than the top-level JSON because TypedDicts used for dtype configs
# are assignable to Mapping[str, object], not to Mapping[str, JSON].
DType = str | int | float | Sequence["DType"] | None | Mapping[str, object]
"""
The widest JSON-like shape that can specify a Zarr data type.

See ``zarr_metadata.dtype.string``, ``.bytes``, and ``.time`` for specific
per-dtype configuration TypedDicts.
"""


__all__ = [
    "DType",
]
