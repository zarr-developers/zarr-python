"""
v2-compatibility chunk key encoding (Zarr v3 core spec).

Intended only to allow existing v2 arrays to be converted to v3 without
having to rename chunks. Not recommended for new arrays.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Final, Literal, NotRequired, TypedDict

V2_CHUNK_KEY_ENCODING_NAME: Final = "v2"
"""The `name` field value of the v2 chunk key encoding."""

V2ChunkKeyEncodingName = Literal["v2"]
"""Literal type of the `name` field of the v2 chunk key encoding."""

V2ChunkKeyEncodingSeparator = Literal["/", "."]
"""Permitted `separator` values for the v2 chunk key encoding.

Defaults to `"."` if absent.
"""


class V2ChunkKeyEncodingConfiguration(TypedDict):
    """Configuration for the v2 chunk key encoding.

    `separator` is optional and defaults to `"."` per spec.
    """

    separator: NotRequired[V2ChunkKeyEncodingSeparator]


class V2ChunkKeyEncodingObject(TypedDict):
    """v2-compatibility chunk key encoding metadata in object form."""

    name: V2ChunkKeyEncodingName
    configuration: NotRequired[V2ChunkKeyEncodingConfiguration]


V2ChunkKeyEncodingMetadata = V2ChunkKeyEncodingObject | V2ChunkKeyEncodingName
"""Permitted JSON shapes for the v2-compatibility chunk-key encoding metadata.

The configuration has no required keys (`separator` defaults to `"."`),
so the short-hand-name form is permitted in addition to the object form.
"""


__all__ = [
    "V2_CHUNK_KEY_ENCODING_NAME",
    "V2ChunkKeyEncodingConfiguration",
    "V2ChunkKeyEncodingMetadata",
    "V2ChunkKeyEncodingName",
    "V2ChunkKeyEncodingObject",
    "V2ChunkKeyEncodingSeparator",
]
