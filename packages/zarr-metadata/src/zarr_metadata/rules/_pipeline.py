"""Codec-pipeline judgments, shared by the array rules and by sharding.

A sharding codec's `codecs` and `index_codecs` are pipelines exactly like
the document's top-level `codecs`, so the ordering and shape checks live
here rather than in either caller: sharding recurses into them at every
nesting depth, and the top-level array rules apply them at depth zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._engine import as_string_mapping, prefixed
from zarr_metadata.v3._shape import entity_name, validate_known_codec_metadata
from zarr_metadata.v3.codec.kind import codec_kind_of_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata.v3.codec.kind import CodecKind

_KIND_RANK: Final = {"array_array": 0, "array_bytes": 1, "bytes_bytes": 2}


def codec_kind(codec: object) -> CodecKind | None:
    """The pipeline kind of `codec`, classified by name alone.

    Spelling-insensitive on purpose: a known codec in an invalid spelling
    still ranks as its kind, so two spellings of the same pipeline always
    get the same ordering verdict and a misspelled known codec is never
    mistaken for an unknown extension (which would suppress the
    exactly-one-`array->bytes` count).
    """
    name = entity_name(codec)
    if name is None:
        return None
    return codec_kind_of_name(name)


def _codec_label(codec: object) -> str:
    if isinstance(codec, str):
        return repr(codec)
    mapping = as_string_mapping(codec)
    if mapping is not None:
        return repr(mapping.get("name"))
    return repr(codec)


def pipeline_order_problems(
    entries: Sequence[object], loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """The spec pipeline shape: `array->array`* `array->bytes` `bytes->bytes`*.

    Codecs of genuinely unknown name are skipped: they impose no ordering
    constraint, and their presence makes the exactly-one-`array->bytes`
    count inconclusive (an unknown codec might be the pipeline's
    `array->bytes` stage), so that check only fires when every codec is
    classified.
    """

    problems: list[ValidationProblem] = []
    kinds = [codec_kind(codec) for codec in entries]
    max_rank_seen = -1
    array_bytes_count = 0
    for index, (codec, kind) in enumerate(zip(entries, kinds, strict=True)):
        if kind is None:
            continue
        rank = _KIND_RANK[kind]
        if rank < max_rank_seen:
            problems.append(
                ValidationProblem(
                    (*loc, index),
                    f"{kind.replace('_', '->')} codec {_codec_label(codec)} may not "
                    "follow a later-stage codec in the pipeline",
                    "invalid_value",
                )
            )
        max_rank_seen = max(max_rank_seen, rank)
        if kind == "array_bytes":
            array_bytes_count += 1
            if array_bytes_count > 1:
                problems.append(
                    ValidationProblem(
                        (*loc, index),
                        f"extra array->bytes codec {_codec_label(codec)}: a pipeline "
                        "has exactly one",
                        "invalid_value",
                    )
                )
    if array_bytes_count == 0 and all(kind is not None for kind in kinds):
        problems.append(
            ValidationProblem(loc, "codec pipeline has no array->bytes codec", "invalid_value")
        )
    return tuple(problems)


def shape_problems(
    entries: Sequence[object], loc: tuple[str | int, ...]
) -> tuple[ValidationProblem, ...]:
    """Shape problems for every known-name codec entry in `entries`.

    Unknown names pass untouched (extension openness); entries without an
    interpretable name decline in favor of the structural validator.
    """
    problems: list[ValidationProblem] = []
    for index, codec in enumerate(entries):
        found = validate_known_codec_metadata(codec)
        # None is "not a known codec" (unjudged); () is "known and valid".
        if found is not None:
            problems.extend(prefixed((*loc, index), found))
    return tuple(problems)


__all__ = [
    "codec_kind",
    "pipeline_order_problems",
    "shape_problems",
]
