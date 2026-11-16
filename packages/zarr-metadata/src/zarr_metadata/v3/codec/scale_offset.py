"""
Scale-offset codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md
"""

from collections.abc import Iterator
from typing import Final

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.definition._definition import (
    AcceptedDeclaration,
    Nested,
    fill_value_problems,
)
from zarr_metadata.v3._definition import (
    ArrayArrayCodecDefinition,
    Chunk,
)
from zarr_metadata.v3._types.codec import (
    SCALE_OFFSET_CODEC_NAME,
    ScaleOffsetCodecConfiguration,
    ScaleOffsetCodecMetadata,
    ScaleOffsetCodecName,
    ScaleOffsetCodecObject,
)
from zarr_metadata.v3.codec._arithmetic import NO_REAL_NUMBERS


def _rules(
    configuration: ScaleOffsetCodecConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """Check that neither scalar is null.

    The registry says each scalar is "JSON-encoded per the input array's
    fill-value rules", and no data type accepts `null` as a fill value.
    Checking which scalars are valid requires the input data type of the
    codec, so a chunk rule does that.
    """
    if "offset" in configuration and configuration["offset"] is None:
        yield ValidationProblem(("offset",), "expected a scalar, got null", "invalid_value")
    if "scale" in configuration and configuration["scale"] is None:
        yield ValidationProblem(("scale",), "expected a scalar, got null", "invalid_value")


def _chunk_rules(
    configuration: ScaleOffsetCodecConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that the input data type is an integer or floating-point type, and that `offset` and `scale` are values of that type.

    The codec "is only defined for data types where multiplication,
    division, addition, and subtraction are well-defined", and of the data
    types its repository defines it supports the integer, unsigned integer
    and floating-point types
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L5-L9),
    so a data type that declares complex or non-numeric values is refused.
    Each of `offset` and `scale` "MUST be encoded to JSON using the
    Zarr V3 fill value encoding for the input array's data type"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L39-L43).
    The rules, not the chunk rules, reject a null.
    """
    source = chunk.data_type
    if not isinstance(source, AcceptedDeclaration):
        return
    if source.definition.number_class in NO_REAL_NUMBERS:
        yield ValidationProblem(
            (),
            f"expected a chunk of an integer or floating-point data type, got {shown(source.name)}",
            "invalid_value",
        )
        return
    for member in ("offset", "scale"):
        value = configuration.get(member)
        if value is not None:
            yield from fill_value_problems(source, value, (member,))


def _transition(
    configuration: ScaleOffsetCodecConfiguration, nested: Nested, chunk: Chunk
) -> Chunk:
    """Return the input chunk unchanged: the values change, but their shape and data type do not.

    Its arithmetic is "performed using the arithmetic semantics of the
    input array's data type"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L47),
    and a later codec must produce any narrower data type
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L153).
    """
    return chunk


SCALE_OFFSET_CODEC: Final = ArrayArrayCodecDefinition(
    name=SCALE_OFFSET_CODEC_NAME,
    configuration=ScaleOffsetCodecConfiguration,
    size="static",
    rules=_rules,
    chunk_rules=_chunk_rules,
    transition=_transition,
)
"""The `scale_offset` codec."""


__all__ = [
    "SCALE_OFFSET_CODEC",
    "SCALE_OFFSET_CODEC_NAME",
    "ScaleOffsetCodecConfiguration",
    "ScaleOffsetCodecMetadata",
    "ScaleOffsetCodecName",
    "ScaleOffsetCodecObject",
]
