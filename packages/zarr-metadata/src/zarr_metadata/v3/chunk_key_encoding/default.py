"""
Default chunk key encoding (Zarr v3 core spec).

The chunk key for a chunk with grid index `(k, j, i, ...)` is formed
by appending `c<sep>k<sep>j<sep>i...` (where `<sep>` is `separator`).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.v3._entity import (
    ChunkKeyEncodingEntity,
)

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


class DefaultChunkKeyEncodingObject(TypedDict, closed=True):
    """Default chunk key encoding metadata in object form."""

    name: DefaultChunkKeyEncodingName
    configuration: NotRequired[DefaultChunkKeyEncodingConfiguration]
    must_understand: NotRequired[bool]


DefaultChunkKeyEncodingMetadata = DefaultChunkKeyEncodingObject | DefaultChunkKeyEncodingName
"""Permitted JSON shapes for the default chunk-key encoding metadata.

The configuration has no required keys (`separator` defaults to `"/"`),
so the short-hand-name form is permitted in addition to the object form.
"""

__all__ = [
    "DEFAULT_CHUNK_KEY_ENCODING_NAME",
    "DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR",
    "DefaultChunkKeyEncoding",
    "DefaultChunkKeyEncodingConfiguration",
    "DefaultChunkKeyEncodingMetadata",
    "DefaultChunkKeyEncodingName",
    "DefaultChunkKeyEncodingObject",
    "DefaultChunkKeyEncodingSeparator",
]


@dataclass(frozen=True)
class DefaultChunkKeyEncoding(ChunkKeyEncodingEntity):
    """The `default` chunk key encoding, coerced from its metadata."""

    separator: DefaultChunkKeyEncodingSeparator | UNSET = UNSET

    identifier: ClassVar[str] = DEFAULT_CHUNK_KEY_ENCODING_NAME

    def to_json(self) -> DefaultChunkKeyEncodingObject | DefaultChunkKeyEncodingName:
        if self.separator is UNSET:
            return "default"
        return {"name": "default", "configuration": {"separator": self.separator}}
