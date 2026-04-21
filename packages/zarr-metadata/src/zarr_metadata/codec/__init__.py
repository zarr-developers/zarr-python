"""
Zarr codec named-config envelope and per-codec configuration types.
"""

from collections.abc import Mapping

Codec = str | Mapping[str, object]
"""
The widest JSON shape that can specify a codec (v2 or v3).

For v3, a codec is a named-config envelope (``{"name": ..., "configuration": ...}``);
for v2, a codec is the numcodecs JSON dict. The accepted-input shape is the
union of both.
"""


__all__ = [
    "Codec",
]
