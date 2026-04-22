"""
Zarr v3 data type spec types.

Each v3 data type has its own submodule:

- Core primitives: `bool`, `int8`/`16`/`32`/`64`, `uint8`/`16`/`32`/`64`,
  `float16`/`32`/`64`, `complex64`/`128`, `raw` (for `r<N>`)
- zarr-extensions: `bytes`, `string`, `numpy_datetime64`, `numpy_timedelta64`,
  `struct`

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from collections.abc import Mapping, Sequence

# Wider than the top-level JSON because TypedDicts used for data type
# configurations are assignable to Mapping[str, object], not to
# Mapping[str, JSON].
DType = str | int | float | Sequence["DType"] | None | Mapping[str, object]
"""The widest JSON-like shape that can specify a Zarr v3 data type."""


__all__ = [
    "DType",
]
