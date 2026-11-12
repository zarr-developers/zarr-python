"""The types of the v3 chunk key encodings.

For each chunk key encoding: the value and `Literal` type of its name,
its configuration and object forms, the JSON shapes its metadata may
take, and the constants that spell its permitted separators.

This module holds types only and imports no definition. Each chunk key
encoding's submodule of `zarr_metadata.v3.chunk_key_encoding` re-exports
its names, and holds the chunk key encoding's definition.
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

# default: https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding

DEFAULT_CHUNK_KEY_ENCODING_NAME: Final = "default"
"""The `name` field value of the default chunk key encoding."""

DefaultChunkKeyEncodingName = Literal["default"]
"""Literal type of the `name` field of the default chunk key encoding."""

DefaultChunkKeyEncodingSeparator = Literal["/", "."]
"""Literal type of permitted `separator` values for the default chunk key encoding.

Defaults to `"/"` if absent.
"""

DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR: Final = ("/", ".")
"""Tuple of permitted values for the `separator` field of the default chunk key encoding."""


class DefaultChunkKeyEncodingConfiguration(TypedDict, closed=True):
    """Configuration for the default chunk key encoding.

    `separator` is optional and defaults to `"/"` per spec.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/default/index.rst#L27-L29
    """

    separator: NotRequired[DefaultChunkKeyEncodingSeparator]
    """The character between the parts of a chunk key: `c/1/23/45` or `c.1.23.45`.

    Absent means `"/"`.
    """


class DefaultChunkKeyEncodingObject(TypedDict, closed=True):
    """Default chunk key encoding metadata in object form."""

    name: DefaultChunkKeyEncodingName
    """The chunk key encoding's name, `"default"`."""
    configuration: NotRequired[DefaultChunkKeyEncodingConfiguration]
    """The chunk key encoding's parameters.

    Absent means every parameter takes its default: a `separator` of `"/"`.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this chunk key encoding must refuse the document.

    Absent means `true`. The spec does not permit `false` for a chunk key encoding
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


DefaultChunkKeyEncodingMetadata = DefaultChunkKeyEncodingObject | DefaultChunkKeyEncodingName
"""Permitted JSON shapes for the default chunk-key encoding metadata.

The configuration has no required keys (`separator` defaults to `"/"`),
so the short-hand-name form is permitted in addition to the object form.
"""


# v2: https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding

V2_CHUNK_KEY_ENCODING_NAME: Final = "v2"
"""The `name` field value of the v2 chunk key encoding."""

V2ChunkKeyEncodingName = Literal["v2"]
"""Literal type of the `name` field of the v2 chunk key encoding."""

V2ChunkKeyEncodingSeparator = Literal["/", "."]
"""Literal type of permitted `separator` values for the v2 chunk key encoding.

Defaults to `"."` if absent.
"""

V2_CHUNK_KEY_ENCODING_SEPARATOR: Final = ("/", ".")
"""Tuple of permitted values for the `separator` field of the v2 chunk key encoding."""


class V2ChunkKeyEncodingConfiguration(TypedDict, closed=True):
    """Configuration for the v2 chunk key encoding.

    `separator` is optional and defaults to `"."` per spec.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/v2/index.rst#L27-L29
    """

    separator: NotRequired[V2ChunkKeyEncodingSeparator]
    """The character between the parts of a chunk key: `1.23.45` or `1/23/45`.

    Absent means `"."`.
    """


class V2ChunkKeyEncodingObject(TypedDict, closed=True):
    """v2-compatibility chunk key encoding metadata in object form."""

    name: V2ChunkKeyEncodingName
    """The chunk key encoding's name, `"v2"`."""
    configuration: NotRequired[V2ChunkKeyEncodingConfiguration]
    """The chunk key encoding's parameters.

    Absent means every parameter takes its default: a `separator` of `"."`.
    """
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this chunk key encoding must refuse the document.

    Absent means `true`. The spec does not permit `false` for a chunk key encoding
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


V2ChunkKeyEncodingMetadata = V2ChunkKeyEncodingObject | V2ChunkKeyEncodingName
"""Permitted JSON shapes for the v2-compatibility chunk-key encoding metadata.

The configuration has no required keys (`separator` defaults to `"."`),
so the short-hand-name form is permitted in addition to the object form.
"""


__all__ = [
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR",
    "V2_CHUNK_KEY_ENCODING_NAME",
    "V2_CHUNK_KEY_ENCODING_SEPARATOR",
    "DefaultChunkKeyEncodingConfiguration",
    "DefaultChunkKeyEncodingMetadata",
    "DefaultChunkKeyEncodingName",
    "DefaultChunkKeyEncodingObject",
    "DefaultChunkKeyEncodingSeparator",
    "V2ChunkKeyEncodingConfiguration",
    "V2ChunkKeyEncodingMetadata",
    "V2ChunkKeyEncodingName",
    "V2ChunkKeyEncodingObject",
    "V2ChunkKeyEncodingSeparator",
]
