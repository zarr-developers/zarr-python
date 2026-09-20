"""
Transpose codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import TypedDict

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    CodecEntity,
    CodecKind,
    MemberTypes,
    is_int,
    problem,
    sequence_of,
)
from zarr_metadata.v3._parts import ArrayParts

TRANSPOSE_CODEC_NAME: Final = "transpose"
"""The `name` field value of the `transpose` codec."""

TransposeCodecName = Literal["transpose"]
"""Literal type of the `name` field of the `transpose` codec."""


class TransposeCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `transpose` codec.

    `order` is a permutation of the dimension indices 0..n-1 that
    specifies the dimension reordering applied during encoding.
    """

    order: tuple[int, ...]


class TransposeCodecObject(TypedDict, closed=True):
    """`transpose` codec metadata in object form."""

    name: TransposeCodecName
    configuration: TransposeCodecConfiguration
    must_understand: NotRequired[bool]


TransposeCodecMetadata = TransposeCodecObject
"""Permitted JSON shape for `transpose` codec metadata.

`order` is required, so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/transpose/index.rst#L60-L66 ("order: Required")
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""

__all__ = [
    "TRANSPOSE_CODEC_NAME",
    "TransposeCodec",
    "TransposeCodecConfiguration",
    "TransposeCodecMetadata",
    "TransposeCodecName",
    "TransposeCodecObject",
]


@dataclass(frozen=True)
class TransposeCodec(CodecEntity):
    """The `transpose` codec, coerced from its metadata."""

    order: tuple[int, ...] = ()

    identifier: ClassVar[str] = TRANSPOSE_CODEC_NAME
    kind: ClassVar[CodecKind] = "array_array"

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {"order": (True, sequence_of(is_int))}

    def problems(self) -> tuple[ValidationProblem, ...]:
        """`order` must permute its own axes.

        Whether it permutes the *array's* axes is a different question --
        it needs the array's rank -- and the rules layer asks that one.
        """
        if sorted(self.order) != list(range(len(self.order))):
            return problem(
                ("order",),
                f"expected a permutation of 0..{len(self.order) - 1}, got {self.order!r}",
                "invalid_value",
            )
        return ()

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """The same array with its axes reordered.

        A transposed regular grid is still a regular grid, so the parts
        survive the trip; the grid metadata does not, because it is no
        longer the grid the document wrote.
        """
        return incoming.with_grid(incoming.grid.permuted(self.order))

    def to_json(self) -> TransposeCodecObject:
        return cast("TransposeCodecObject", super().to_json())
