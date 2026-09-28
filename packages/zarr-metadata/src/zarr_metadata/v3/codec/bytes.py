"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
    Nested,
    storage_of,
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


def _chunk_rules(
    configuration: BytesCodecConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """An `endian` for a data type whose values take several bytes, and a data type of fixed size.

    `endian` is "Required for data types for which endianness is
    applicable"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L64-L69):
    every core type of numbers but `bool`, `int8` and `uint8`, whose
    values take one byte and do "not depend on endian"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L84-L101).
    Raw bits `r8` take one byte too; of wider raw bits the spec does not
    yet say.
    The codec "encodes arrays of fixed-size numeric data types"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L28-L30),
    and a type whose values vary in size takes a codec of its own:
    `string` "is only compatible with the vlen-utf8 codec"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md?plain=1#L25).
    Asked of what the data type says of its storage; unknown, it is left
    be.
    """
    if chunk.data_type is None:
        return
    storage = storage_of(chunk.data_type)
    written = chunk.data_type.name
    if storage == "multi_byte" and "endian" not in configuration:
        yield ValidationProblem(
            ("endian",),
            f"expected an endian, since each {written!r} value takes several bytes",
            "missing_key",
        )
    elif storage == "variable_length":
        yield ValidationProblem(
            (),
            f"expected a data type of fixed size, got {shown(written)}, whose values vary in size",
            "invalid_value",
        )


BYTES_CODEC: Final = CodecDefinition(
    name=BYTES_CODEC_NAME,
    configuration=BytesCodecConfiguration,
    kind="array_bytes",
    size="static",
    chunk_rules=_chunk_rules,
)
"""The `bytes` codec."""

__all__ = [
    "BYTES_CODEC",
    "BYTES_CODEC_NAME",
    "ENDIANNESS",
    "BytesCodecConfiguration",
    "BytesCodecMetadata",
    "BytesCodecName",
    "BytesCodecObject",
    "Endianness",
]
