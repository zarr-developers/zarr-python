"""
Cast-value codec types.

See https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3._common import MetadataField

CAST_VALUE_CODEC_NAME: Final = "cast_value"
"""The `name` field value of the `cast_value` codec."""

CastValueCodecName = Literal["cast_value"]
"""Literal type of the `name` field of the `cast_value` codec."""

RoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]
"""Permitted values for the `rounding` configuration field.

Defaults to `"nearest-even"` if absent.
"""

OutOfRangeMode = Literal["clamp", "wrap"]
"""Permitted values for the `out_of_range` configuration field.

If absent, out-of-range values are an encoding/decoding error.
"""

ScalarMapEntry = tuple[object, object]
"""A single `[input, output]` mapping in a `scalar_map` direction.

Each scalar is JSON-encoded per its data type's fill-value rules (so
e.g. `"NaN"` and `"+Infinity"` are permitted).
"""


class ScalarMap(TypedDict):
    """Optional encode/decode scalar overrides for the cast_value codec."""

    encode: NotRequired[tuple[ScalarMapEntry, ...]]
    decode: NotRequired[tuple[ScalarMapEntry, ...]]


class CastValueCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `cast_value` codec.

    `data_type` is the target data type that input values are cast to. It
    is the same shape as the top-level array `data_type` field: either a
    bare-string primitive name or a `{name, configuration}` envelope.
    """

    data_type: MetadataField
    rounding: NotRequired[RoundingMode]
    out_of_range: NotRequired[OutOfRangeMode]
    scalar_map: NotRequired[ScalarMap]


class CastValueCodecObject(TypedDict):
    """`cast_value` codec metadata in object form."""

    name: CastValueCodecName
    configuration: CastValueCodecConfiguration


CastValueCodecMetadata = CastValueCodecObject
"""Permitted JSON shape for `cast_value` codec metadata.

`configuration.data_type` is required, so only the object form is valid;
the short-hand-name form is not permitted by the spec for this codec.
"""


__all__ = [
    "CAST_VALUE_CODEC_NAME",
    "CastValueCodecConfiguration",
    "CastValueCodecMetadata",
    "CastValueCodecName",
    "CastValueCodecObject",
    "OutOfRangeMode",
    "RoundingMode",
    "ScalarMap",
    "ScalarMapEntry",
]
