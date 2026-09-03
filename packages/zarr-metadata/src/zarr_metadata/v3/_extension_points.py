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

from typing import Final, Literal

from zarr_metadata.v3.data_type.raw import RAW_BYTES_NAME_PATTERN

ExtensionPointField = Literal[
    "data_type", "chunk_grid", "chunk_key_encoding", "codecs", "storage_transformers"
]
"""The v3 array metadata fields whose values name an extension."""

DATA_TYPE: Final[ExtensionPointField] = "data_type"
CHUNK_GRID: Final[ExtensionPointField] = "chunk_grid"
CHUNK_KEY_ENCODING: Final[ExtensionPointField] = "chunk_key_encoding"
CODECS: Final[ExtensionPointField] = "codecs"
STORAGE_TRANSFORMERS: Final[ExtensionPointField] = "storage_transformers"

RAW_BYTES_FAMILY: Final = "r<N>"
"""Canonical key for the parameterized raw-bytes data type family.

Spelled as the spec writes the family; the angle brackets keep it
unforgeable by a real name.
"""


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
    "STORAGE_TRANSFORMERS",
    "ExtensionPointField",
    "canonical_name",
]
