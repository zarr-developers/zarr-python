"""
Scale-offset codec types.

See https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/scale_offset
"""

from typing import Final, Literal, NotRequired, TypedDict

SCALE_OFFSET_CODEC_NAME: Final = "scale_offset"
"""The `name` field value of the `scale_offset` codec."""

ScaleOffsetCodecName = Literal["scale_offset"]
"""Literal type of the `name` field of the `scale_offset` codec."""


class ScaleOffsetCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `scale_offset` codec.

    Both fields are optional. A missing `offset` is the additive identity
    (e.g. 0 for numeric types); a missing `scale` is the multiplicative
    identity (e.g. 1). Each scalar is JSON-encoded per the input array's
    fill-value rules, so `"NaN"` and `"+Infinity"` style strings are
    permitted in addition to numbers.
    """

    offset: NotRequired[object]
    scale: NotRequired[object]


class ScaleOffsetCodecObject(TypedDict):
    """`scale_offset` codec metadata in object form.

    `configuration` is itself optional per spec — when both `offset` and
    `scale` are at their identity defaults, the codec is a no-op and the
    entire `configuration` field may be omitted.
    """

    name: ScaleOffsetCodecName
    configuration: NotRequired[ScaleOffsetCodecConfiguration]


ScaleOffsetCodecMetadata = ScaleOffsetCodecObject | ScaleOffsetCodecName
"""Permitted JSON shapes for `scale_offset` codec metadata.

The configuration has no required keys (both `offset` and `scale` are
optional, and the configuration itself is optional), so the short-hand-name
form is permitted in addition to the object form.
"""


__all__ = [
    "SCALE_OFFSET_CODEC_NAME",
    "ScaleOffsetCodecConfiguration",
    "ScaleOffsetCodecMetadata",
    "ScaleOffsetCodecName",
    "ScaleOffsetCodecObject",
]
