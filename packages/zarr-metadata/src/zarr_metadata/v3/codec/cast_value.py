"""
Cast-value codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, Self

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._entity import (
    ArrayArrayCodec,
    Configuration,
    DataTypeEntity,
    Opaque,
)
from zarr_metadata.v3._parts import ArrayParts

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

Each scalar is JSON-encoded per its data type's fill-value rules (so
e.g. `"NaN"` and `"+Infinity"` are permitted).
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

    data_type: ZarrV3MetadataFieldJSON
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


__all__ = [
    "CAST_OUT_OF_RANGE_MODE",
    "CAST_ROUNDING_MODE",
    "CAST_VALUE_CODEC_NAME",
    "SCALAR_MAP_KEYS",
    "CastOutOfRangeMode",
    "CastRoundingMode",
    "CastValueCodec",
    "CastValueCodecConfiguration",
    "CastValueCodecMetadata",
    "CastValueCodecName",
    "CastValueCodecObject",
    "CastValueOptions",
    "ScalarMap",
    "ScalarMapEntry",
]


SCALAR_MAP_KEYS: Final = ("encode", "decode")
"""The two directions a `scalar_map` can override, both optional."""


@dataclass(frozen=True)
class CastValueOptions(Configuration):
    """What `cast_value` is configured with."""

    data_type: DataTypeEntity | Opaque
    rounding: CastRoundingMode | UNSET = UNSET
    out_of_range: CastOutOfRangeMode | UNSET = UNSET
    scalar_map: ScalarMap | UNSET = UNSET


@dataclass(frozen=True)
class CastValueCodec(ArrayArrayCodec):
    """The `cast_value` codec, coerced from its metadata.

    Holds the data type it casts to, so like `sharding_indexed` it is
    read in a scope rather than on its own.
    """

    configuration: CastValueOptions

    identifier: ClassVar[str] = CAST_VALUE_CODEC_NAME
    variable_size: ClassVar[bool] = False

    def canonical(self) -> Self:
        """The target data type in its own canonical form."""
        return self.with_configuration(data_type=self.configuration.data_type.canonical())

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """The same parts, holding the type this codec casts to."""
        data_type = self.configuration.data_type
        return incoming.with_data_type(data_type if isinstance(data_type, DataTypeEntity) else None)
