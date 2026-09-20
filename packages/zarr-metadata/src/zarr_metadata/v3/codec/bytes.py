"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import TypedDict

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    CodecEntity,
    CodecKind,
    DataTypeEntity,
    MemberTypes,
    one_of,
    problem,
)
from zarr_metadata.v3._parts import ArrayParts

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
class BytesCodec(CodecEntity):
    """The `bytes` codec, coerced from its metadata.

    `endian` is optional and absent means something: a one-byte data type
    has no byte order to state, and the spec lets such an array omit it.
    """

    endian: Endianness | None = None

    identifier: ClassVar[str] = BYTES_CODEC_NAME
    kind: ClassVar[CodecKind] = "array_bytes"

    member_types: ClassVar[MemberTypes] = {"endian": (False, one_of(ENDIANNESS))}

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """The data type reaching here must have a raw byte representation.

        A variable-length type has no fixed one, so this codec cannot
        encode it. A multi-byte one has several orderings, so `endian` is
        required -- and the message names the type, because inside a
        shard's `index_codecs` the array is the shard index, whose
        `uint64` type appears nowhere in the document.
        """
        data_type = incoming.data_type if incoming is not None else None
        if not isinstance(data_type, DataTypeEntity):
            return ()
        storage = data_type.storage_class()
        name = type(data_type).identifier
        if storage == "variable_length":
            return problem(
                (),
                f"bytes codec is not compatible with variable-length data_type {name!r}",
                "invalid_value",
            )
        if storage == "multi_byte" and self.endian is None:
            return problem(
                ("endian",),
                f"endian is required for data type {name!r}, which contains multi-byte values",
                "missing_key",
            )
        return ()

    def to_json(self) -> BytesCodecObject | BytesCodecName:
        return cast("BytesCodecObject | BytesCodecName", super().to_json())
