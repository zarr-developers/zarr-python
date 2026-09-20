"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import TypedDict

from zarr_metadata.v3._entity import (
    CodecKind,
    MemberTypes,
    MetadataEntity,
    one_of,
)

BYTES_CODEC_NAME: Final = "bytes"
"""The `name` field value of the `bytes` codec."""

BytesCodecName = Literal["bytes"]
"""Literal type of the `name` field of the `bytes` codec."""

Endianness = Literal["little", "big"]
"""Literal type of byte order of multi-byte numeric data."""

ENDIANNESS: Final = ("little", "big")
"""Tuple of permitted values for the `endian` field of the `bytes` codec."""


class BytesCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `bytes` codec.

    The `endian` field is required for multi-byte data types.
    """

    endian: NotRequired[Endianness]


class BytesCodecObject(TypedDict, closed=True):
    """`bytes` codec metadata in object form.

    `configuration` is itself optional — when no configuration fields are
    set, the entire `configuration` key may be omitted. This matches the
    bare-string short-hand form (`BytesCodecName`) at the canonical data
    level; both encodings describe a `bytes` codec with default settings.
    """

    name: BytesCodecName
    configuration: NotRequired[BytesCodecConfiguration]
    must_understand: NotRequired[bool]


BytesCodecMetadata = BytesCodecObject | BytesCodecName
"""Permitted JSON shapes for `bytes` codec metadata.

The configuration has no required keys (`endian` is conditionally required
at runtime based on data type), so the spec's short-hand-name form is
permitted in addition to the object form, and the object form may itself
omit `configuration` entirely.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L64-L69 ("endian: Required for data types for which endianness is applicable")
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""

__all__ = [
    "BYTES_CODEC_NAME",
    "ENDIANNESS",
    "BytesCodec",
    "BytesCodecConfiguration",
    "BytesCodecMetadata",
    "BytesCodecName",
    "BytesCodecObject",
    "Endianness",
]


@dataclass(frozen=True)
class BytesCodec(MetadataEntity):
    """The `bytes` codec, coerced from its metadata.

    `endian` is optional and absent means something: a one-byte data type
    has no byte order to state, and the spec lets such an array omit it.
    """

    endian: Endianness | None = None

    identifier: ClassVar[str] = BYTES_CODEC_NAME
    kind: ClassVar[CodecKind] = "array_bytes"

    member_types: ClassVar[MemberTypes] = {"endian": (False, one_of(ENDIANNESS))}

    def to_json(self) -> BytesCodecObject | BytesCodecName:
        return cast("BytesCodecObject | BytesCodecName", super().to_json())
