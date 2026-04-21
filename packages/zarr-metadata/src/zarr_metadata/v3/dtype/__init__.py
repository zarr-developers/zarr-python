"""
Zarr data type spec types.
"""

from collections.abc import Mapping, Sequence

# Wider than the top-level JSON because TypedDicts used for dtype configs
# are assignable to Mapping[str, object], not to Mapping[str, JSON].
DType = str | int | float | Sequence["DType"] | None | Mapping[str, object]
"""
The widest JSON-like shape that can specify a Zarr data type.

See the submodules for specific per-dtype types:

- `zarr_metadata.v3.dtype.primitive` -- core v3 primitives (bool, int*, uint*, float*, complex*)
- `zarr_metadata.v3.dtype.bytes`     -- `bytes`, `null_terminated_bytes`
- `zarr_metadata.v3.dtype.string`    -- `string`, `fixed_length_utf32`
- `zarr_metadata.v3.dtype.time`      -- `numpy.datetime64`, `numpy.timedelta64`
- `zarr_metadata.v3.dtype.struct`    -- `struct`
"""


__all__ = [
    "DType",
]
