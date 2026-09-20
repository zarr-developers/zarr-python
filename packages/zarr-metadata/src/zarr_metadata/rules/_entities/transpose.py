"""Composition rules and spec transition for the `transpose` codec.

Whether `order` is a permutation of its own indices is a fact about the
value, checked by `v3._shape`. What is left here needs the array that
reached the codec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._chunk_grid import ChunkGrid
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.rules._spec import ArrayParts, spec_transition
from zarr_metadata.v3._extension_points import CODECS
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

_ARRAY_V3 = "zarr_v3_array"


@spec_transition(TRANSPOSE_CODEC_NAME)
def permute_grid(configuration: Mapping[str, object], incoming: ArrayParts) -> ArrayParts:
    """The outgoing grid is the incoming one with its axes reordered.

    A transposed grid is still a grid, so the parts survive the codec with
    their lengths permuted. An order that is not a permutation of the rank
    yields a grid of unknown extents — the rules below report the order
    itself, and extents derived from a bad order would be a guess.
    """
    order = cast("tuple[int, ...]", configuration["order"])
    if sorted(order) != list(range(len(order))):
        return incoming.with_grid(ChunkGrid(incoming.grid.rank, None))
    return incoming.with_grid(incoming.grid.permuted(order))


@entity_rule(_ARRAY_V3, CODECS, TRANSPOSE_CODEC_NAME, reads=frozenset({"order"}))
def order_matches_incoming_rank(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArrayParts | None
) -> tuple[ValidationProblem, ...]:
    """A transpose permutes the array it receives, so ranks must agree.

    Judged against what actually reaches this codec, not the document's
    `shape`: inside a shard that is the inner chunk, and after another
    transpose it is that transpose's output. Declines when the rank is
    unknown.
    """
    rank = incoming.grid.rank if incoming is not None else None
    if rank is None:
        return ()
    order = cast("tuple[int, ...]", configuration["order"])
    if len(order) == rank:
        return ()
    return (
        ValidationProblem(
            ("order",),
            f"order has {len(order)} entries but the incoming array has {rank} dimensions",
            "invalid_value",
        ),
    )
