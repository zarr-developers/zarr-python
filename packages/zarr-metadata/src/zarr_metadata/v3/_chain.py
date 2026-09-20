"""Walking a codec pipeline, handing each codec what reaches it.

The spec orders a pipeline `array->array`* `array->bytes` `bytes->bytes`*,
and each array-to-array codec transforms what the next one sees. So a
codec's configuration is judged against the array that *reaches* it, not
against the document's top-level fields: `transpose` permutes the grid,
`cast_value` changes the element type, and a shard that follows either
one sees the transformed array.

This is where the walk lives rather than in `zarr_metadata.rules`,
because a `sharding_indexed` codec holds two pipelines of its own and has
to walk them to judge itself.

Propagation stops -- every later codec receives `None` -- at the
array-to-bytes boundary, where there is no array left, and after any
codec that cannot say what it does to the array: one whose name is out of
scope, or a modelled one with no transition. An unknown codec might
change anything, so declining is the only honest answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import CodecEntity, within

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata.v3._entity import Loc
    from zarr_metadata.v3._parts import ArrayParts

_KIND_RANK = {"array_array": 0, "array_bytes": 1, "bytes_bytes": 2}


def _label(codec: object) -> str:
    if isinstance(codec, CodecEntity):
        return repr(type(codec).identifier)
    return repr(codec)


def order_problems(codecs: Sequence[object], loc: Loc) -> tuple[ValidationProblem, ...]:
    """Whether the pipeline is shaped the way the spec orders it.

    A codec out of scope is skipped: it imposes no ordering constraint,
    and it makes the exactly-one-`array->bytes` count inconclusive,
    because it might be the pipeline's own `array->bytes` stage. So that
    count is only checked when every codec is in scope.
    """
    problems: list[ValidationProblem] = []
    latest = -1
    array_bytes = 0
    for index, codec in enumerate(codecs):
        if not isinstance(codec, CodecEntity):
            continue
        kind = type(codec).kind
        rank = _KIND_RANK[kind]
        if rank < latest:
            problems.append(
                ValidationProblem(
                    (*loc, index),
                    f"{kind.replace('_', '->')} codec {_label(codec)} may not "
                    "follow a later-stage codec in the pipeline",
                    "invalid_value",
                )
            )
        latest = max(latest, rank)
        if kind == "array_bytes":
            array_bytes += 1
            if array_bytes > 1:
                problems.append(
                    ValidationProblem(
                        (*loc, index),
                        f"extra array->bytes codec {_label(codec)}: a pipeline has exactly one",
                        "invalid_value",
                    )
                )
    if array_bytes == 0 and all(isinstance(codec, CodecEntity) for codec in codecs):
        problems.append(
            ValidationProblem(loc, "codec pipeline has no array->bytes codec", "invalid_value")
        )
    return tuple(problems)


def chain_problems(
    codecs: Sequence[object], start: ArrayParts | None, loc: Loc
) -> tuple[ValidationProblem, ...]:
    """Every problem this pipeline has, ordering and per-codec alike.

    `start` is what the first codec receives: the document's own array, or
    a shard's inner chunk, or its index.
    """
    problems = list(order_problems(codecs, loc))
    incoming = start
    for index, codec in enumerate(codecs):
        if not isinstance(codec, CodecEntity):
            # Out of scope: unjudged, and everything after it is too.
            incoming = None
            continue
        problems.extend(within((*loc, index), codec.incoming_problems(incoming)))
        incoming = (
            None
            if incoming is None or type(codec).kind != "array_array"
            else codec.transition(incoming)
        )
    return tuple(problems)


__all__ = [
    "chain_problems",
    "order_problems",
]
