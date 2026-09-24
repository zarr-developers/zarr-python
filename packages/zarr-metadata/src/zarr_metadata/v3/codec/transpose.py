"""
Transpose codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import CodecDefinition

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


def _rules(configuration: TransposeCodecConfiguration) -> Iterator[ValidationProblem]:
    """`order` permutes its own axes.

    Whether it permutes the *array's* axes needs the array's rank, and is
    asked where the codec meets the array.
    """
    order = configuration["order"]
    if sorted(order) != list(range(len(order))):
        yield ValidationProblem(
            ("order",),
            f"expected a permutation of 0..{len(order) - 1}, got {order!r}",
            "invalid_value",
        )


TRANSPOSE_CODEC: Final = CodecDefinition(
    name=TRANSPOSE_CODEC_NAME,
    configuration=TransposeCodecConfiguration,
    kind="array_array",
    size="static",
    rules=_rules,
)
"""The `transpose` codec."""


__all__ = [
    "TRANSPOSE_CODEC",
    "TRANSPOSE_CODEC_NAME",
    "TransposeCodecConfiguration",
    "TransposeCodecMetadata",
    "TransposeCodecName",
    "TransposeCodecObject",
]
