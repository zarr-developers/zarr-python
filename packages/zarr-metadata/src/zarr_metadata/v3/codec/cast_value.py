"""
Cast-value codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
    DataTypeField,
    Nested,
    Read,
    fill_value_problems,
)
from zarr_metadata.v3.codec._arithmetic import COMPLEX, FLOATING_POINT, NOT_NUMBERS

CAST_VALUE_CODEC_NAME: Final = "cast_value"
"""The `name` field value of the `cast_value` codec."""

CastValueCodecName = Literal["cast_value"]
"""Literal type of the `name` field of the `cast_value` codec."""

CastRoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]
"""Literal type of permitted values for the `rounding` configuration field.

Defaults to `"nearest-even"` if absent.
"""

CAST_ROUNDING_MODE: Final = (
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
)
"""Tuple of permitted values for the `rounding` field of the `cast_value` codec."""

CastOutOfRangeMode = Literal["clamp", "wrap"]
"""Literal type of permitted values for the `out_of_range` configuration field.

If absent, out-of-range values are an encoding/decoding error.
"""

CAST_OUT_OF_RANGE_MODE: Final = ("clamp", "wrap")
"""Tuple of permitted values for the `out_of_range` field of the `cast_value` codec."""

ScalarMapEntry = tuple[JSONValue, JSONValue]
"""A single `[input, output]` mapping in a `scalar_map` direction.

Each scalar is a fill value of its data type: a float's infinities are
`"Infinity"` and `"-Infinity"`, as the core encoding writes them, though
the codec's own example writes `"+Infinity"`.
"""


class ScalarMap(TypedDict, closed=True):
    """Optional encode/decode scalar overrides for the cast_value codec."""

    encode: NotRequired[tuple[ScalarMapEntry, ...]]
    decode: NotRequired[tuple[ScalarMapEntry, ...]]


class CastValueCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `cast_value` codec.

    `data_type` is the target data type that input values are cast to. It
    is the same shape as the top-level array `data_type` field: either a
    bare-string primitive name or a `{name, configuration}` envelope.
    """

    data_type: DataTypeField
    rounding: NotRequired[CastRoundingMode]
    out_of_range: NotRequired[CastOutOfRangeMode]
    scalar_map: NotRequired[ScalarMap]


class CastValueCodecObject(TypedDict, closed=True):
    """`cast_value` codec metadata in object form."""

    name: CastValueCodecName
    configuration: CastValueCodecConfiguration
    must_understand: NotRequired[bool]


CastValueCodecMetadata = CastValueCodecObject
"""Permitted JSON shape for `cast_value` codec metadata.

`configuration.data_type` is required, so only the object form is valid;
the short-hand-name form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md#L33-L36 and #L46-L48 (required fields)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""


_NO_REAL_NUMBERS: Final = NOT_NUMBERS | COMPLEX
"""The data types the codec takes neither from nor to: none of them models real numbers."""


def _scalars(
    configuration: CastValueCodecConfiguration, side: Literal["source", "target"]
) -> Iterator[tuple[tuple[str | int, ...], JSONValue]]:
    """Each scalar of `scalar_map` written in the data type on one side of the cast, with where it sits.

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
    """It casts to a data type that models real numbers, `wrap`s only to an integral one, and maps to values of it.

    The codec "is only defined for data types that model real numbers"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9);
    `wrap` is "Only permitted when data_type is an integral type"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L84);
    and a scalar of the target type is encoded with its fill value
    encoding. Judged of the data type it casts to, which the scope read;
    of the one it is handed, the chunk rules judge the rest. A target that
    models no real numbers is the one problem reported of it.
    """
    target = nested.get(("data_type",))
    if not isinstance(target, Read):
        return
    name, written = target.definition.name, target.name
    if name in _NO_REAL_NUMBERS:
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
    """It is handed a data type that models real numbers, and maps from values of it.

    "The same ordered procedure applies during decoding, with input and
    output data types swapped"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L13),
    so the data type it is handed is held to what the one it casts to is.
    """
    source = chunk.data_type
    if not isinstance(source, Read):
        return
    if source.definition.name in _NO_REAL_NUMBERS:
        yield ValidationProblem(
            (),
            f"expected a chunk of a data type that models real numbers, got {shown(source.name)}",
            "invalid_value",
        )
        return
    for at, scalar in _scalars(configuration, "source"):
        yield from fill_value_problems(source, scalar, at)


def _transition(configuration: CastValueCodecConfiguration, nested: Nested, chunk: Chunk) -> Chunk:
    """The chunk's values cast to the data type it names, and nothing else changed.

    It "converts (casts) the values of the input array to a new data
    type ... and it leaves all other array properties intact"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L3).
    It hands on the data type field as the scope read it, which says
    nothing of the values when the scope did not read it.
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

The data type it casts to is a nested field, read in the scope the codec
is read in, and the data type of the chunk it hands on.
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
