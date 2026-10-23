"""
Transpose codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
)

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


def _rules(
    configuration: TransposeCodecConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """Check that `order` is a permutation of its own indices.

    A chunk rule checks whether `order` matches the axes of the input
    chunk.
    """
    order = configuration["order"]
    if sorted(order) != list(range(len(order))):
        yield ValidationProblem(
            ("order",),
            f"expected a permutation of 0..{len(order) - 1}, got {shown(order)}",
            "invalid_value",
        )


def _chunk_rules(
    configuration: TransposeCodecConfiguration, nested: Nested, chunk: Chunk
) -> Iterator[ValidationProblem]:
    """Check that `order` has one entry for each axis of the input chunk.

    "a permutation of 0, 1, ..., n-1, where n is the number of dimensions
    in the decoded chunk representation provided as input to this codec"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/transpose/index.rst#L63-L66).
    """
    order = configuration["order"]
    if chunk.rank is not None and len(order) != chunk.rank:
        yield ValidationProblem(
            ("order",),
            f"expected {chunk.rank} entries, one per axis of the chunk the codec is "
            f"handed, got {len(order)}",
            "invalid_value",
        )


def _transition(configuration: TransposeCodecConfiguration, nested: Nested, chunk: Chunk) -> Chunk:
    """Return the chunk with its axes reordered by `order`. The data type does not change.

    "B_shape[i] = A_shape[order[i]]", where A is the input chunk and B is
    the output chunk, with "the same data type as A"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/transpose/index.rst#L73-L79).
    If `order` has a different number of axes than the chunk, the output
    lengths are unknown.
    """
    lengths = chunk.lengths
    order = configuration["order"]
    if lengths is None or len(order) != len(lengths):
        return Chunk(None, chunk.data_type)
    return Chunk(tuple(lengths[axis] for axis in order), chunk.data_type)


TRANSPOSE_CODEC: Final = CodecDefinition(
    name=TRANSPOSE_CODEC_NAME,
    configuration=TransposeCodecConfiguration,
    kind="array_array",
    size="static",
    rules=_rules,
    chunk_rules=_chunk_rules,
    transition=_transition,
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
