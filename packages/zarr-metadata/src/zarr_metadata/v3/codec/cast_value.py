"""
Cast-value codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md
"""

from collections.abc import Iterator
from typing import Final, Literal

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.definition._definition import (
    AcceptedDeclaration,
    Nested,
    fill_value_problems,
)
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
)
from zarr_metadata.v3._types.codec import (
    CAST_OUT_OF_RANGE_MODE,
    CAST_ROUNDING_MODE,
    CAST_VALUE_CODEC_NAME,
    CastOutOfRangeMode,
    CastRoundingMode,
    CastValueCodecConfiguration,
    CastValueCodecMetadata,
    CastValueCodecName,
    CastValueCodecObject,
    ScalarMap,
    ScalarMapEntry,
)
from zarr_metadata.v3.codec._arithmetic import FLOATING_POINT, NO_REAL_NUMBERS


def _scalars(
    configuration: CastValueCodecConfiguration, side: Literal["source", "target"]
) -> Iterator[tuple[tuple[str | int, ...], JSONValue]]:
    """Yield each scalar of `scalar_map` that is written in the data type of one side of the cast, with its location.

    "For encode, the input data type is the array data type before
    casting and the output data type is the target data_type. For decode,
    the input data type is the target data_type and the output data type
    is the array data type before casting"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L107-L109).
    """
    scalar_map = configuration.get("scalar_map")
    if scalar_map is None:
        return
    for direction, entries in (
        ("encode", scalar_map.get("encode", ())),
        ("decode", scalar_map.get("decode", ())),
    ):
        at = int((direction == "encode") == (side == "target"))
        for index, entry in enumerate(entries):
            yield ("scalar_map", direction, index, at), entry[at]


def _rules(
    configuration: CastValueCodecConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """Check that the target data type models real numbers, that `wrap` has an integral target, and that mapped values are valid in the target.

    The codec "is only defined for data types that model real numbers"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9);
    `wrap` is "Only permitted when data_type is an integral type"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L84);
    and a scalar of the target type is encoded with its fill value
    encoding. These rules check the target data type, which the scope
    read. The chunk rules check the input data type. If the target does
    not model real numbers, that is the only problem reported for it.
    """
    target = nested.get(("data_type",))
    if not isinstance(target, AcceptedDeclaration):
        return
    name, written = target.definition.name, target.name
    if name in NO_REAL_NUMBERS:
        yield ValidationProblem(
            ("data_type",),
            f"expected a data type that models real numbers, got {shown(written)}",
            "invalid_value",
        )
        return
    if configuration.get("out_of_range") == "wrap" and name in FLOATING_POINT:
        yield ValidationProblem(
            ("out_of_range",),
            f"expected an integral data_type to wrap to, got {shown(written)}",
            "invalid_value",
        )
    for at, scalar in _scalars(configuration, "target"):
        yield from fill_value_problems(target, scalar, at)


def _chunk_rules(
    configuration: CastValueCodecConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that the input data type models real numbers, and that mapped values are valid in it.

    "The same ordered procedure applies during decoding, with input and
    output data types swapped"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L13),
    so the input data type must meet the same requirements as the target
    data type.
    """
    source = chunk.data_type
    if not isinstance(source, AcceptedDeclaration):
        return
    if source.definition.name in NO_REAL_NUMBERS:
        yield ValidationProblem(
            (),
            f"expected a chunk of a data type that models real numbers, got {shown(source.name)}",
            "invalid_value",
        )
        return
    for at, scalar in _scalars(configuration, "source"):
        yield from fill_value_problems(source, scalar, at)


def _transition(configuration: CastValueCodecConfiguration, nested: Nested, chunk: Chunk) -> Chunk:
    """Return the chunk with its values cast to the target data type. Nothing else changes.

    It "converts (casts) the values of the input array to a new data
    type ... and it leaves all other array properties intact"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L3).
    The output data type is the data type declaration as the scope read it. If
    the scope did not read that declaration, the output data type is unknown.
    """
    return Chunk(chunk.lengths, nested.get(("data_type",)))


CAST_VALUE_CODEC: Final = CodecDefinition(
    name=CAST_VALUE_CODEC_NAME,
    configuration=CastValueCodecConfiguration,
    kind="array_array",
    size="static",
    rules=_rules,
    chunk_rules=_chunk_rules,
    transition=_transition,
)
"""The `cast_value` codec.

The target data type is a nested declaration. It is read in the same scope as
the codec, and it becomes the data type of the output chunk.
"""


__all__ = [
    "CAST_OUT_OF_RANGE_MODE",
    "CAST_ROUNDING_MODE",
    "CAST_VALUE_CODEC",
    "CAST_VALUE_CODEC_NAME",
    "CastOutOfRangeMode",
    "CastRoundingMode",
    "CastValueCodecConfiguration",
    "CastValueCodecMetadata",
    "CastValueCodecName",
    "CastValueCodecObject",
    "ScalarMap",
    "ScalarMapEntry",
]
