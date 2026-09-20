"""The Zarr v3 extension points, and how names are keyed under them.

Names are unique only within an extension point (`bytes` is both a core
codec and a registered data type), so every table in this package is
keyed by `(field, canonical name)`.

`canonical_name` is identity except for raw-byte data types: every `r<N>`
spelling, valid or not, maps to `RAW_BYTES_FAMILY`, so a malformed member
of that family is reported as a misspelling rather than passing as an
unknown extension. Canonical names are lookup keys and are never emitted.
"""

from __future__ import annotations

from zarr_metadata.v3._entity import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    ExtensionPointField,
)
from zarr_metadata.v3.data_type.raw import RAW_BYTES_FAMILY, RAW_BYTES_NAME_PATTERN


def canonical_name(field: ExtensionPointField, name: str) -> str:
    """`name` reduced to the key this package tables it under."""
    if field == DATA_TYPE and RAW_BYTES_NAME_PATTERN.fullmatch(name) is not None:
        return RAW_BYTES_FAMILY
    return name


__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "DATA_TYPE",
    "RAW_BYTES_FAMILY",
    "ExtensionPointField",
    "canonical_name",
]
