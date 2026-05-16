"""
Default chunk key encoding (Zarr v3 core spec).

The chunk key for a chunk with grid index `(k, j, i, ...)` is formed
by appending `c<sep>k<sep>j<sep>i...` (where `<sep>` is `separator`).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

DEFAULT_CHUNK_KEY_ENCODING_NAME: Final = "default"
"""The `name` field value of the default chunk key encoding."""

DefaultChunkKeyEncodingName = Literal["default"]
"""Literal type of the `name` field of the default chunk key encoding."""

DefaultChunkKeyEncodingSeparator = Literal["/", "."]
"""Permitted `separator` values for the default chunk key encoding.

Defaults to `"/"` if absent.
"""


class DefaultChunkKeyEncodingConfiguration(TypedDict):
    """Configuration for the default chunk key encoding.

    `separator` is optional and defaults to `"/"` per spec.
    """

    separator: NotRequired[DefaultChunkKeyEncodingSeparator]


class DefaultChunkKeyEncodingObject(TypedDict):
    """Default chunk key encoding metadata in object form."""

    name: DefaultChunkKeyEncodingName
    configuration: NotRequired[DefaultChunkKeyEncodingConfiguration]


DefaultChunkKeyEncodingMetadata = DefaultChunkKeyEncodingObject | DefaultChunkKeyEncodingName
"""Permitted JSON shapes for the default chunk-key encoding metadata.

The configuration has no required keys (`separator` defaults to `"/"`),
so the short-hand-name form is permitted in addition to the object form.
"""

__all__ = [
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DefaultChunkKeyEncodingConfiguration",
    "DefaultChunkKeyEncodingMetadata",
    "DefaultChunkKeyEncodingName",
    "DefaultChunkKeyEncodingObject",
    "DefaultChunkKeyEncodingSeparator",
]
