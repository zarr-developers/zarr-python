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
    Configured,
)
from zarr_metadata.v3._parts import ArrayParts

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
class ScaleOffsetCodec(ArrayArrayCodec, Configured):
    """The `scale_offset` codec, coerced from its metadata.

    Both members are optional and any JSON scalar is well-typed here; what
    a given value means depends on the data type it is applied to, which
    is a question for the rules layer.
    """

    configuration: ScaleOffsetOptions

    identifier: ClassVar[str] = SCALE_OFFSET_CODEC_NAME
    variable_size: ClassVar[bool] = False

    @property
    def offset(self) -> JSONValue | UNSET:
        return self.configuration.offset

    @property
    def scale(self) -> JSONValue | UNSET:
        return self.configuration.scale

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """The same array, element for element.

        The registry entry removed the `astype` field, so this codec no
        longer changes the element type -- only the values.
        """
        return incoming
