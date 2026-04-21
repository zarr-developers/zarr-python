"""
Zarr codec metadata types.
"""

from collections.abc import Mapping

Codec = str | Mapping[str, object]
"""
The widest JSON shape that can specify a codec (v2 or v3).

For v3, a codec is a `{"name": ..., "configuration": ...}` mapping (or
a bare `str` shorthand); for v2, a codec is the numcodecs JSON dict.
The accepted-input shape is the union of both.
"""


__all__ = [
    "Codec",
]
