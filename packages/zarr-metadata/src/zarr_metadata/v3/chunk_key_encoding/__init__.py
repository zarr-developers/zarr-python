"""
Zarr v3 chunk key encoding metadata types.

Each chunk key encoding lives in its own submodule:

  - `default` -- v3 default encoding (`/`-separated)
  - `v2`      -- v2-compatibility encoding (`.`-separated by default)

Both are defined by the v3 core spec.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Literal

ChunkKeySeparator = Literal["/", "."]
"""The two permitted chunk key separator characters per the v3 core spec."""


__all__ = [
    "ChunkKeySeparator",
]
