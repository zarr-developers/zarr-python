"""
Scale-offset codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import CodecDefinition

SCALE_OFFSET_CODEC_NAME: Final = "scale_offset"
"""The `name` field value of the `scale_offset` codec."""

ScaleOffsetCodecName = Literal["scale_offset"]
"""Literal type of the `name` field of the `scale_offset` codec."""


class ScaleOffsetCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `scale_offset` codec.

    Both fields are optional. A missing `offset` is the additive identity
    (e.g. 0 for numeric types); a missing `scale` is the multiplicative
    identity (e.g. 1). Each scalar is JSON-encoded per the input array's
    fill-value rules, so `"NaN"` and `"+Infinity"` style strings are
    permitted in addition to numbers.
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


def _rules(configuration: ScaleOffsetCodecConfiguration) -> Iterator[ValidationProblem]:
    """Neither scalar is null.

    The registry says each is "JSON-encoded per the input array's
    fill-value rules", and no data type admits `null` as a fill value.
    Which scalar it must be needs the data type the codec is handed, and
    is asked where the codec meets the array.
    """
    if "offset" in configuration and configuration["offset"] is None:
        yield ValidationProblem(("offset",), "expected a scalar, got null", "invalid_value")
    if "scale" in configuration and configuration["scale"] is None:
        yield ValidationProblem(("scale",), "expected a scalar, got null", "invalid_value")


SCALE_OFFSET_CODEC: Final = CodecDefinition(
    name=SCALE_OFFSET_CODEC_NAME,
    configuration=ScaleOffsetCodecConfiguration,
    kind="array_array",
    size="static",
    rules=_rules,
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
