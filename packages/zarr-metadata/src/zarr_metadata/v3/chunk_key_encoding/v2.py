"""
v2-compatibility chunk key encoding (Zarr v3 core spec).

Intended only to allow existing v2 arrays to be converted to v3 without
having to rename chunks. Not recommended for new arrays.

Naming note: these are Zarr **v3** types. The leading `V2` in
`V2ChunkKeyEncodingMetadata` (and friends) is the encoding's registered
*entity name* (`"v2"`), not the format-version marker that `ZarrV2...`
names carry — this package's version-prefixed names always spell it
`ZarrV2` / `ZarrV3`.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Final

from zarr_metadata.v3._definition import ChunkKeyEncodingDefinition
from zarr_metadata.v3._types.chunk_key_encoding import (
    V2_CHUNK_KEY_ENCODING_NAME,
    V2_CHUNK_KEY_ENCODING_SEPARATOR,
    V2ChunkKeyEncodingConfiguration,
    V2ChunkKeyEncodingMetadata,
    V2ChunkKeyEncodingName,
    V2ChunkKeyEncodingObject,
    V2ChunkKeyEncodingSeparator,
)


def _canonical(configuration: V2ChunkKeyEncodingConfiguration) -> V2ChunkKeyEncodingConfiguration:
    """Remove a `separator` of `.`, because an absent `separator` means the same thing.

    "If not specified, `separator` defaults to `.`"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/v2/index.rst#L27-L29),
    so the two spellings describe the same encoding.
    """
    if configuration.get("separator") != ".":
        return configuration
    return {}


V2_CHUNK_KEY_ENCODING: Final = ChunkKeyEncodingDefinition(
    name=V2_CHUNK_KEY_ENCODING_NAME,
    configuration=V2ChunkKeyEncodingConfiguration,
    canonical=_canonical,
)
"""The `v2` chunk key encoding. The type of `separator` already restricts its values, so the encoding has no rules of its own."""


__all__ = [
    "V2_CHUNK_KEY_ENCODING",
    "V2_CHUNK_KEY_ENCODING_NAME",
    "V2_CHUNK_KEY_ENCODING_SEPARATOR",
    "V2ChunkKeyEncodingConfiguration",
    "V2ChunkKeyEncodingMetadata",
    "V2ChunkKeyEncodingName",
    "V2ChunkKeyEncodingObject",
    "V2ChunkKeyEncodingSeparator",
]
