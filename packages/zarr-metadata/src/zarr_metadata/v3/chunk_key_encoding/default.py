"""
Default chunk key encoding (Zarr v3 core spec).

The chunk key for a chunk with grid index `(k, j, i, ...)` is formed
by appending `c<sep>k<sep>j<sep>i...` (where `<sep>` is `separator`).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Final

from zarr_metadata.v3._definition import ChunkKeyEncodingDefinition
from zarr_metadata.v3._types.chunk_key_encoding import (
    DEFAULT_CHUNK_KEY_ENCODING_NAME,
    DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR,
    DefaultChunkKeyEncodingConfiguration,
    DefaultChunkKeyEncodingMetadata,
    DefaultChunkKeyEncodingName,
    DefaultChunkKeyEncodingObject,
    DefaultChunkKeyEncodingSeparator,
)


def _canonical(
    configuration: DefaultChunkKeyEncodingConfiguration,
) -> DefaultChunkKeyEncodingConfiguration:
    """Remove a `separator` of `/`, because an absent `separator` means the same thing.

    "If not specified, `separator` defaults to `/`"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/default/index.rst#L27-L29),
    so the two spellings describe the same encoding.
    """
    if configuration.get("separator") != "/":
        return configuration
    return {}


DEFAULT_CHUNK_KEY_ENCODING: Final = ChunkKeyEncodingDefinition(
    name=DEFAULT_CHUNK_KEY_ENCODING_NAME,
    configuration=DefaultChunkKeyEncodingConfiguration,
    canonical=_canonical,
)
"""The `default` chunk key encoding. The type of `separator` already restricts its values, so the encoding has no rules of its own."""


__all__ = [
    "DEFAULT_CHUNK_KEY_ENCODING",
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR",
    "DefaultChunkKeyEncodingConfiguration",
    "DefaultChunkKeyEncodingMetadata",
    "DefaultChunkKeyEncodingName",
    "DefaultChunkKeyEncodingObject",
    "DefaultChunkKeyEncodingSeparator",
]
