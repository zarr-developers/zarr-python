"""The Zarr v3 extension points.

Names are unique only within an extension point (`bytes` is both a core
codec and a registered data type), so every table in this package is
keyed by `(field, name)`.

A name that no key matches may still belong to a *family* -- one class
covering many spellings, as the raw-byte data types cover every `r<N>`.
`Context.resolve` asks each entity through `accepts`, so a family is
registered like anything else and this module holds no table of
spellings.
"""

from __future__ import annotations

from zarr_metadata.v3._entity import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    STORAGE_TRANSFORMERS,
    ExtensionPointField,
)
from zarr_metadata.v3.data_type.raw import RAW_BYTES_FAMILY

__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "DATA_TYPE",
    "RAW_BYTES_FAMILY",
    "STORAGE_TRANSFORMERS",
    "ExtensionPointField",
]
