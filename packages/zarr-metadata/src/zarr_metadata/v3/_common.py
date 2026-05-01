"""Internal cross-cutting aliases for Zarr v3 metadata.

This module is private (underscore-prefixed) and exists to avoid circular
imports between leaf modules and sub-package `__init__.py` re-exports.
Public consumers should import `MetadataField` from `zarr_metadata.v3`.
"""

from zarr_metadata._common import NamedConfig

MetadataField = str | NamedConfig
"""The JSON shape of any v3 metadata extension-point entry: either a bare
short-hand name string or a `{name, configuration}` envelope.

Used for `data_type`, `chunk_grid`, `chunk_key_encoding`, individual
codec entries, and `storage_transformers` in v3 array metadata, and for
the inner `codecs` / `index_codecs` lists of the `sharding_indexed`
codec.
"""


__all__ = [
    "MetadataField",
]
