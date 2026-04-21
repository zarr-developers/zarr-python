"""
V2-compatibility chunk key encoding (Zarr v3 core spec).

Intended only to allow existing v2 arrays to be converted to v3 without
having to rename chunks. Not recommended for new arrays.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Final, Literal, NotRequired, TypedDict

from zarr_metadata.v3.chunk_key_encoding import ChunkKeySeparator

V2_CHUNK_KEY_ENCODING_NAME: Final = "v2"
"""The `name` field value of the v2 chunk key encoding."""

V2ChunkKeyEncodingName = Literal["v2"]
"""Literal type of the `name` field of the v2 chunk key encoding."""


class V2ChunkKeyEncodingConfiguration(TypedDict):
    """Configuration for the v2 chunk key encoding.

    `separator` is optional and defaults to `"."` per spec.
    """

    separator: NotRequired[ChunkKeySeparator]


class V2ChunkKeyEncoding(TypedDict):
    """V2-compatibility chunk key encoding metadata."""

    name: V2ChunkKeyEncodingName
    configuration: NotRequired[V2ChunkKeyEncodingConfiguration]


__all__ = [
    "V2_CHUNK_KEY_ENCODING_NAME",
    "V2ChunkKeyEncoding",
    "V2ChunkKeyEncodingConfiguration",
    "V2ChunkKeyEncodingName",
]
