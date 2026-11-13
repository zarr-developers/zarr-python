"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from collections.abc import Iterator
from typing import Final

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import (
    ArrayBytesCodecDefinition,
    Chunk,
    storage_of,
)
from zarr_metadata.v3._types.codec import (
    BYTES_CODEC_NAME,
    ENDIANNESS,
    BytesCodecConfiguration,
    BytesCodecMetadata,
    BytesCodecName,
    BytesCodecObject,
    Endianness,
)


def _chunk_rules(
    configuration: BytesCodecConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that `endian` is present for a data type whose values take several bytes, and that the data type has a fixed size.

    `endian` is "Required for data types for which endianness is
    applicable"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L64-L69):
    this is every core numeric type except `bool`, `int8` and `uint8`.
    The values of those types take one byte and do "not depend on endian"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L84-L101).
    Raw bits `r8` also take one byte. The spec does not yet say anything
    about wider raw bits.
    The codec "encodes arrays of fixed-size numeric data types"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/bytes/index.rst#L28-L30),
    and a type whose values vary in size needs its own codec. For
    example, `string` "is only compatible with the vlen-utf8 codec"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md?plain=1#L25).
    These checks use the storage that the data type declares. If the
    storage is unknown, nothing is checked.
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


BYTES_CODEC: Final = ArrayBytesCodecDefinition(
    name=BYTES_CODEC_NAME,
    configuration=BytesCodecConfiguration,
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
