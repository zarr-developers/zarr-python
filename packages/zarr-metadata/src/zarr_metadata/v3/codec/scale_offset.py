"""
Scale-offset codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.definition._definition import (
    AcceptedField,
    Nested,
    fill_value_problems,
)
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
)
from zarr_metadata.v3.codec._arithmetic import NOT_NUMBERS

SCALE_OFFSET_CODEC_NAME: Final = "scale_offset"
"""The `name` field value of the `scale_offset` codec."""

ScaleOffsetCodecName = Literal["scale_offset"]
"""Literal type of the `name` field of the `scale_offset` codec."""


class ScaleOffsetCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `scale_offset` codec.

    Both fields are optional. A missing `offset` is the additive identity
    (e.g. 0 for numeric types); a missing `scale` is the multiplicative
    identity (e.g. 1). Each scalar is encoded as a fill value of the input
    data type of the codec: an integer for an integer type, and for a
    float type a number, `"NaN"`, `"Infinity"`, `"-Infinity"` or a hex
    string.
    """

    offset: NotRequired[JSONValue]
    scale: NotRequired[JSONValue]


class ScaleOffsetCodecObject(TypedDict, closed=True):
    """`scale_offset` codec metadata in object form.

    `configuration` is itself optional per spec — when both `offset` and
    `scale` are at their identity defaults, the codec is a no-op and the
    entire `configuration` field may be omitted.
      https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md#L18 and #L35
    """

    name: ScaleOffsetCodecName
    configuration: NotRequired[ScaleOffsetCodecConfiguration]
    must_understand: NotRequired[bool]


ScaleOffsetCodecMetadata = ScaleOffsetCodecObject | ScaleOffsetCodecName
"""Permitted JSON shapes for `scale_offset` codec metadata.

The configuration has no required keys (both `offset` and `scale` are
optional, and the configuration itself is optional), so the short-hand-name
form is permitted in addition to the object form.
"""


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
    """Check that the input data type supports arithmetic, and that `offset` and `scale` are values of that type.

    The codec "is only defined for data types where multiplication,
    division, addition, and subtraction are well-defined"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L5-L9),
    and each of `offset` and `scale` "MUST be encoded to JSON using the
    Zarr V3 fill value encoding for the input array's data type"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L39-L43).
    The rules, not the chunk rules, reject a null.
    """
    source = chunk.data_type
    if not isinstance(source, AcceptedField):
        return
    if source.definition.name in NOT_NUMBERS:
        yield ValidationProblem(
            (),
            f"expected a chunk of a data type with arithmetic, got {shown(source.name)}",
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


SCALE_OFFSET_CODEC: Final = CodecDefinition(
    name=SCALE_OFFSET_CODEC_NAME,
    configuration=ScaleOffsetCodecConfiguration,
    kind="array_array",
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
