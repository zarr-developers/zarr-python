"""Composition rules and spec transition for the `transpose` codec."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.rules._spec import ArraySpec, spec_transition
from zarr_metadata.v3._extension_points import CODECS
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

_ARRAY_V3 = "zarr_v3_array"


@spec_transition(TRANSPOSE_CODEC_NAME)
def permute_shape(configuration: Mapping[str, object], incoming: ArraySpec) -> ArraySpec:
    """The outgoing shape is the incoming shape permuted by `order`.

    Declines (shape None) when the order is not a permutation of the
    incoming rank: the rules below report that, and any shape derived
    from a bad order would be a guess.
    """
    order = cast("tuple[int, ...]", configuration["order"])
    shape = incoming.shape
    if shape is None or sorted(order) != list(range(len(shape))):
        return incoming.with_shape(None)
    return incoming.with_shape(tuple(shape[axis] for axis in order))


@entity_rule(_ARRAY_V3, CODECS, TRANSPOSE_CODEC_NAME)
def order_is_a_permutation(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """`order` must be a permutation of its own indices.

    Checked without reference to the incoming shape, so it holds even
    when propagation has stopped upstream.
    """
    order = cast("tuple[int, ...]", configuration["order"])
    if sorted(order) == list(range(len(order))):
        return ()
    return (
        ValidationProblem(
            ("order",),
            f"expected a permutation of 0..{len(order) - 1}, got {order!r}",
            "invalid_value",
        ),
    )


@entity_rule(_ARRAY_V3, CODECS, TRANSPOSE_CODEC_NAME)
def order_matches_incoming_rank(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """A transpose permutes the array it receives, so ranks must agree.

    Judged against the *incoming* spec, not the document's `shape`: inside
    a shard the incoming array is the inner chunk, and after another
    transpose it is that transpose's output. Declines when the incoming
    shape is unknown.
    """
    if incoming.shape is None:
        return ()
    order = cast("tuple[int, ...]", configuration["order"])
    if len(order) == len(incoming.shape):
        return ()
    return (
        ValidationProblem(
            ("order",),
            f"order has {len(order)} entries but the incoming array has "
            f"{len(incoming.shape)} dimensions",
            "invalid_value",
        ),
    )
