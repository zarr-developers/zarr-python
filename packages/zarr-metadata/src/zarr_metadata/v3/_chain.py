"""Walking a codec pipeline, handing each codec what reaches it.

The spec orders a pipeline `array->array`* `array->bytes` `bytes->bytes`*,
and each array-to-array codec transforms what the next one sees. So a
codec's configuration is judged against the array that *reaches* it, not
against the document's top-level fields: `transpose` permutes the grid,
`cast_value` changes the element type, and a shard that follows either
one sees the transformed array.

The walk produces the resolved pipeline: at each position the codec and
the array that reaches it, and for a codec that holds pipelines of its
own -- a shard's inner chunks and its index -- those pipelines refined
the same way. Validation is what the walk finds on the way.

Propagation stops -- every later codec receives `None` -- at the
array-to-bytes boundary, where there is no array left, and after any
codec that cannot say what it does to the array: one whose name is out of
scope, or a modelled one with no transition. An unknown codec might
change anything, so declining is the only honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import ArrayArrayCodec, ArrayBytesCodec, CodecEntity, within

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr_metadata.v3._entity import Loc, Opaque
    from zarr_metadata.v3._parts import ArrayParts


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """One position of a refined pipeline: the codec, and the array that reaches it."""

    codec: CodecEntity | Opaque
    incoming: ArrayParts | None
    """What reaches this codec.

    None past the array->bytes boundary, where there is no array, and
    after a codec that could not say what it does to one.
    """
    inner: Mapping[str, Pipeline]
    """The pipelines this codec holds, refined, by the member that holds each; empty for most."""


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A codec pipeline refined against the array handed to it: what reaches each codec."""

    stages: tuple[PipelineStage, ...]


def _stage(codec: CodecEntity) -> tuple[int, str]:
    """Where in the pipeline a codec stands, as a rank and as the spec names it."""
    if isinstance(codec, ArrayArrayCodec):
        return 0, "array->array"
    if isinstance(codec, ArrayBytesCodec):
        return 1, "array->bytes"
    return 2, "bytes->bytes"


def _label(codec: CodecEntity | Opaque) -> str:
    if isinstance(codec, CodecEntity):
        return repr(type(codec).identifier)
    return repr(codec.json)


def order_problems(
    codecs: Sequence[CodecEntity | Opaque], loc: Loc
) -> tuple[ValidationProblem, ...]:
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
        rank, stage = _stage(codec)
        if rank < latest:
            problems.append(
                ValidationProblem(
                    (*loc, index),
                    f"{stage} codec {_label(codec)} may not "
                    "follow a later-stage codec in the pipeline",
                    "invalid_value",
                )
            )
        latest = max(latest, rank)
        if isinstance(codec, ArrayBytesCodec):
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


def refine_pipeline(
    codecs: Sequence[CodecEntity | Opaque], start: ArrayParts | None, loc: Loc
) -> tuple[Pipeline, tuple[ValidationProblem, ...]]:
    """The pipeline refined against `start`, with every problem found on the way.

    `start` is what the first codec receives: the document's own array,
    or a shard's inner chunk, or its index. Each codec is asked what it
    cannot take of what reaches it, then what it hands on; a codec that
    holds pipelines has them refined in turn, located under the member
    that holds each. Ordering is judged first, over the whole pipeline.
    """
    problems = list(order_problems(codecs, loc))
    stages: list[PipelineStage] = []
    incoming = start
    for index, codec in enumerate(codecs):
        if not isinstance(codec, CodecEntity):
            # Out of scope: unjudged, and everything after it is too.
            stages.append(PipelineStage(codec, incoming, {}))
            incoming = None
            continue
        problems.extend(within((*loc, index), codec.incoming_problems(incoming)))
        inner: dict[str, Pipeline] = {}
        for member, (held, handed) in codec.inner_pipelines(incoming).items():
            pipeline, found = refine_pipeline(held, handed, (*loc, index, "configuration", member))
            inner[member] = pipeline
            problems.extend(found)
        stages.append(PipelineStage(codec, incoming, inner))
        incoming = (
            codec.transition(incoming)
            if incoming is not None and isinstance(codec, ArrayArrayCodec)
            else None
        )
    return Pipeline(tuple(stages)), tuple(problems)


__all__ = [
    "Pipeline",
    "PipelineStage",
    "order_problems",
    "refine_pipeline",
]
