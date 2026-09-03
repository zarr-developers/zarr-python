"""Composition rules for the `sharding_indexed` codec.

Sharding is the one entity whose configuration contains whole pipelines
and its own geometry, so its rules recurse: the inner `codecs` and
`index_codecs` are judged by the same pipeline checks that judge the
document's top-level `codecs`, at every nesting depth.

Every geometry judgment here is against the *incoming* array spec — the
array as transformed by every codec before this one — never against the
document's chunk grid directly. A `transpose` in front of a shard changes
which extents the shard has to divide, and reading the grid instead gave
wrong verdicts in both directions: it accepted an inner chunk that did
not divide the transposed shape and rejected one that did.

The inner pipeline receives the inner chunk as its incoming spec (with
the incoming data type carried through), so a transpose or nested shard
inside it is judged against the inner chunk, recursively — each sharding
level encloses the next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._pipeline import pipeline_order_problems, shape_problems
from zarr_metadata.rules._registry import entity_rule, run_chain_rules
from zarr_metadata.rules._spec import NOTHING_KNOWN, ArraySpec
from zarr_metadata.v3._extension_points import CODECS
from zarr_metadata.v3._shape import entity_name
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC_NAME
from zarr_metadata.v3.codec.gzip import GZIP_CODEC_NAME
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC_NAME
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

_ARRAY_V3 = "zarr_v3_array"
_VARIABLE_SIZE_CODECS = frozenset(
    {BLOSC_CODEC_NAME, GZIP_CODEC_NAME, SHARDING_INDEXED_CODEC_NAME, ZSTD_CODEC_NAME}
)


@entity_rule(_ARRAY_V3, CODECS, SHARDING_INDEXED_CODEC_NAME)
def inner_chunk_extents_are_positive(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
    return tuple(
        ValidationProblem(
            ("chunk_shape", position),
            f"expected a positive chunk extent, got {extent}",
            "invalid_value",
        )
        for position, extent in enumerate(chunk_shape)
        if extent < 1
    )


@entity_rule(_ARRAY_V3, CODECS, SHARDING_INDEXED_CODEC_NAME)
def inner_chunks_tile_the_incoming_array(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """The inner chunk must rank-match and evenly divide the array it receives.

    Declines when the incoming shape is unknown — an unclassified codec
    upstream, or a non-regular grid at the top level — rather than
    guessing from the document.
    """
    if incoming.shape is None:
        return ()
    outer = incoming.shape
    inner = cast("tuple[int, ...]", configuration["chunk_shape"])
    if len(inner) != len(outer):
        return (
            ValidationProblem(
                ("chunk_shape",),
                f"chunk_shape has {len(inner)} entries but the incoming array has "
                f"{len(outer)} dimensions",
                "invalid_value",
            ),
        )
    return tuple(
        ValidationProblem(
            ("chunk_shape", position),
            f"inner chunk extent {inner_extent} does not evenly divide the "
            f"incoming extent {outer_extent}",
            "invalid_value",
        )
        for position, (outer_extent, inner_extent) in enumerate(zip(outer, inner, strict=True))
        if inner_extent >= 1 and outer_extent % inner_extent != 0
    )


@entity_rule(_ARRAY_V3, CODECS, SHARDING_INDEXED_CODEC_NAME)
def inner_pipelines_are_pipelines(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """`codecs` and `index_codecs` obey the pipeline rules, recursively.

    Both get the ordering and shape judgments the top-level pipeline gets,
    plus the entity rules of whatever codecs appear inside. The inner
    `codecs` chain starts from the inner chunk with the incoming data
    type; a nested shard or transpose inside it is therefore judged
    against the inner chunk, and its own transitions carry on from there.
    The `index_codecs` chain encodes the shard index, a `uint64` array
    whose shape this package does not compute.
    """
    inner_shape = configuration["chunk_shape"]
    if not isinstance(inner_shape, tuple) or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 1
        for v in cast("tuple[object, ...]", inner_shape)
    ):
        inner_start = NOTHING_KNOWN
    else:
        # The inner pipeline encodes the inner chunk: same type as arrived
        # here, shape of one inner chunk.
        inner_start = incoming.with_shape(cast("tuple[int, ...]", inner_shape))
    problems: list[ValidationProblem] = []
    for key in ("codecs", "index_codecs"):
        entries = configuration[key]
        if not isinstance(entries, (list, tuple)):
            continue
        sequence = cast("tuple[object, ...]", entries)
        problems.extend(pipeline_order_problems(sequence, (key,)))
        problems.extend(shape_problems(sequence, (key,)))
        # The index pipeline encodes the shard index, not the array: a
        # uint64 array of offsets and lengths, so e.g. the bytes codec
        # inside it still needs an endianness.
        start = inner_start if key == "codecs" else ArraySpec(None, "uint64")
        problems.extend(run_chain_rules(CODECS, sequence, document, (key,), start))
    return tuple(problems)


@entity_rule(_ARRAY_V3, CODECS, SHARDING_INDEXED_CODEC_NAME)
def index_codecs_have_fixed_encoded_size(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """The shard index must have an encoded size derivable from metadata."""
    entries = cast("tuple[object, ...]", configuration["index_codecs"])
    return tuple(
        ValidationProblem(
            ("index_codecs", index),
            f"{name!r} produces variable-size output; index_codecs must be fixed-size",
            "invalid_value",
        )
        for index, entry in enumerate(entries)
        if (name := entity_name(entry)) in _VARIABLE_SIZE_CODECS
    )
