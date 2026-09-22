"""
Scale-offset codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    ArrayArrayCodec,
    Configuration,
    DataTypeEntity,
    problem,
)
from zarr_metadata.v3._parts import ArrayParts
from zarr_metadata.v3.data_type._families import FloatDataType, IntegerDataType

if TYPE_CHECKING:
    from collections.abc import Iterator

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

__all__ = [
    "SCALE_OFFSET_CODEC_NAME",
    "ScaleOffsetCodec",
    "ScaleOffsetCodecConfiguration",
    "ScaleOffsetCodecMetadata",
    "ScaleOffsetCodecName",
    "ScaleOffsetCodecObject",
    "ScaleOffsetOptions",
]


@dataclass(frozen=True)
class ScaleOffsetOptions(Configuration):
    """What `scale_offset` is configured with."""

    offset: JSONValue | UNSET = UNSET
    scale: JSONValue | UNSET = UNSET

    def problems(self) -> "Iterator[ValidationProblem]":
        """Each value is a scalar of the array's type, so neither is null.

        The registry says each is "JSON-encoded per the input array's
        fill-value rules", and no data type admits `null` as a fill value.
        Which scalar it should be needs the data type, so that part is the
        document's question, not this codec's.
        """
        if self.offset is None:
            yield ValidationProblem(("offset",), "expected a scalar, got null", "invalid_value")
        if self.scale is None:
            yield ValidationProblem(("scale",), "expected a scalar, got null", "invalid_value")


@dataclass(frozen=True)
class ScaleOffsetCodec(ArrayArrayCodec):
    """The `scale_offset` codec, coerced from its metadata.

    Both members are optional and any JSON scalar is well-typed here; what
    a given value means depends on the data type it is applied to, which
    `incoming_problems` asks of the type that reaches the codec.
    """

    configuration: ScaleOffsetOptions

    identifier: ClassVar[str] = SCALE_OFFSET_CODEC_NAME
    variable_size: ClassVar[bool] = False

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """What the array handed to this codec must be, and what its members must be for it.

        The registry defines the codec for data types with arithmetic and
        lists the integer and floating-point ones. `offset` and `scale`
        are each "encoded to JSON using the Zarr V3 fill value encoding
        for the input array's data type" -- the type that reaches this
        codec, which after a `cast_value` is not the array's own -- so
        each is a fill value of that type, and that type judges it: the
        string `"0"` is no `float32` and no `int32`.
        """
        data_type = incoming.data_type if incoming is not None else None
        if not isinstance(data_type, DataTypeEntity):
            return ()
        if not isinstance(data_type, (IntegerDataType, FloatDataType)):
            return problem(
                (),
                "scale_offset is defined for integer and floating-point data types, not "
                f"{type(data_type).identifier!r}",
                "invalid_value",
            )
        return tuple(
            found
            for member, value in (
                ("offset", self.configuration.offset),
                ("scale", self.configuration.scale),
            )
            if value is not UNSET
            for found in data_type.fill_value_problems(value, (member,))
        )

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """The same array, element for element.

        The registry entry removed the `astype` field, so this codec no
        longer changes the element type -- only the values.
        """
        return incoming
