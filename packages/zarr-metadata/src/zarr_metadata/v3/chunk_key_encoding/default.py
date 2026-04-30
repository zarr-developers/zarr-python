"""
Default chunk key encoding (Zarr v3 core spec).

The chunk key for a chunk with grid index `(k, j, i, ...)` is formed
by appending `c<sep>k<sep>j<sep>i...` (where `<sep>` is `separator`).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Final, Literal, NotRequired, TypedDict

from zarr_metadata.v3.chunk_key_encoding import ChunkKeySeparator

DEFAULT_CHUNK_KEY_ENCODING_NAME: Final = "default"
"""The `name` field value of the default chunk key encoding."""

DefaultChunkKeyEncodingName = Literal["default"]
"""Literal type of the `name` field of the default chunk key encoding."""


class DefaultChunkKeyEncodingConfiguration(TypedDict):
    """Configuration for the default chunk key encoding.

    `separator` is optional and defaults to `"/"` per spec.
    """

    separator: NotRequired[ChunkKeySeparator]


class DefaultChunkKeyEncoding(TypedDict):
    """Default chunk key encoding metadata."""

    name: DefaultChunkKeyEncodingName
    configuration: NotRequired[DefaultChunkKeyEncodingConfiguration]


__all__ = [
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DefaultChunkKeyEncoding",
    "DefaultChunkKeyEncodingConfiguration",
    "DefaultChunkKeyEncodingName",
]
