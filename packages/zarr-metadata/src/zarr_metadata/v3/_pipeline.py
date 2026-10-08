"""A codec pipeline, read: each codec with the chunk it is handed.

A pipeline is array -> array codecs, then one array -> bytes codec, then
bytes -> bytes codecs. The array hands its first codec a `Chunk` -- "the
same data type as the Zarr array, and shape equal to the chunk shape"
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1048-L1050),
the lengths its grid's chunks take along each axis -- and each array ->
array codec hands the next the chunk its `transition` says.
Each codec handed an array is judged by its chunk rules against the chunk
it is handed: a `bytes` codec handed a multi-byte data type without an
`endian`, a `transpose` whose `order` has another number of axes. A codec
that holds pipelines of its own has each read the same way, handed the
chunk its `pipelines` says: a shard's inner codecs its inner chunks, its
index codecs the shard index.

Nothing is guessed. A codec the scope did not read -- nothing in scope
claims it, or it has a problem of its own -- might do anything, so the
codec after it is handed a chunk nothing is known of, as is the codec
after one that says nothing of what it hands on; a codec after that
hands on only what it says of its own accord, as `cast_value` names its
data type. One nothing in scope claims might be of any of the three
kinds, so the order is judged without it. A chunk rule judges what is
known of its chunk and leaves the rest.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypeGuard, cast

from zarr_metadata._json import ValidationProblem, is_object, with_input
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
    CodecKind,
    Read,
    Resolved,
    asked,
    read_only,
    ruled,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v3._definition import Nested, Problems


def _no_stages() -> Mapping[str, tuple[Stage, ...]]:
    """The pipelines of a codec that holds none, or was not read: none."""
    return {}


@dataclass(frozen=True, slots=True)
class Stage:
    """One codec of a pipeline, and the chunk it is handed."""

    codec: Resolved[CodecDefinition[Any]]
    """The codec, as the scope read it."""
    incoming: Chunk | None
    """The chunk it is handed.

    None for a codec handed bytes -- a bytes -> bytes codec, or any codec
    after the array -> bytes codec -- and for a codec nothing in scope
    claims when what it is handed is not known to be an array.
    """
    inner: Mapping[str, tuple[Stage, ...]] = dataclasses.field(default_factory=_no_stages)
    """The pipelines it holds, by the member of its configuration that holds each: each codec with the chunk it is handed."""


_POSITIONS: Final[Mapping[CodecKind, int]] = {
    "array_array": 0,
    "array_bytes": 1,
    "bytes_bytes": 2,
}
"""Where each kind of codec comes in a pipeline."""

_SPOKEN: Final[Mapping[CodecKind, str]] = {
    "array_array": "array -> array",
    "array_bytes": "array -> bytes",
    "bytes_bytes": "bytes -> bytes",
}


def read_pipeline(
    codecs: Sequence[Resolved[CodecDefinition[Any]]], chunk: Chunk, loc: Loc = ()
) -> tuple[tuple[Stage, ...], Problems]:
    """Each of `codecs`, codec fields a scope read, as a pipeline handed `chunk`, with the chunk it is handed, and what is wrong with them.

    The order first: array -> array codecs, then one array -> bytes codec,
    then bytes -> bytes codecs. Then each codec in turn, handed the chunk
    the one before it handed on: judged by its chunk rules, which locate
    their problems in its configuration; each pipeline it holds read the
    same way, where it sits in the configuration; and, an array -> array
    codec, asked what it hands on. Problems are located under `loc`, where
    the codecs sit, each codec at its index.
    """
    problems = list(_order_problems(codecs, loc))
    stages: list[Stage] = []
    # What the next codec is handed: None once that is not known to be an
    # array -- past the array -> bytes codec, or a codec of unknown kind.
    handed: Chunk | None = chunk
    for index, codec in enumerate(codecs):
        definition = codec.definition
        if definition is None:
            stages.append(Stage(codec, handed))
            handed = None
            continue
        if definition.kind == "bytes_bytes":
            stages.append(Stage(codec, None))
            handed = None
            continue
        incoming = Chunk() if handed is None else handed
        at = (*loc, index, "configuration")
        inner = _no_stages()
        if isinstance(codec, Read):
            configuration, nested = codec.configuration, codec.nested
            problems.extend(_chunk_problems(definition, configuration, nested, incoming, at))
            inner, found = _inner_pipelines(definition, configuration, nested, incoming, at)
            problems.extend(found)
        stages.append(Stage(codec, incoming, inner))
        if definition.kind == "array_bytes":
            handed = None
        elif not isinstance(codec, Read):
            handed = Chunk()
        else:
            handed = _handed_on(definition, codec.configuration, codec.nested, incoming, at)
    if len(problems) == 0:
        return tuple(stages), ()
    return tuple(stages), with_input(problems, tuple(codec.json for codec in codecs), loc)


def _order_problems(
    codecs: Sequence[Resolved[CodecDefinition[Any]]], loc: Loc
) -> Iterator[ValidationProblem]:
    """Array -> array codecs, then one array -> bytes codec, then bytes -> bytes codecs.

    "the list of codecs must be of the following form: zero or more
    array -> array codecs; followed by exactly one array -> bytes codec;
    followed by zero or more bytes -> bytes codecs"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L974-L983).
    A codec nothing in scope claims is left out: it might be any of the
    three, so a pipeline that holds one is not refused for holding no
    array -> bytes codec. A codec out of place is reported where it sits,
    and so is an array -> bytes codec after the first.
    """
    furthest: CodecDefinition[Any] | None = None
    encoder: CodecDefinition[Any] | None = None
    unclaimed = False
    for index, codec in enumerate(codecs):
        definition = codec.definition
        if definition is None:
            unclaimed = True
            continue
        if furthest is not None and _POSITIONS[definition.kind] < _POSITIONS[furthest.kind]:
            yield ValidationProblem(
                (*loc, index),
                f"expected {_SPOKEN[definition.kind]} codecs before "
                f"{_SPOKEN[furthest.kind]} codecs, got {definition.name!r} after "
                f"{furthest.name!r}",
                "invalid_value",
            )
        elif definition.kind == "array_bytes" and encoder is not None:
            yield ValidationProblem(
                (*loc, index),
                f"expected one array -> bytes codec, got {definition.name!r} after "
                f"{encoder.name!r}",
                "invalid_value",
            )
        if furthest is None or _POSITIONS[definition.kind] > _POSITIONS[furthest.kind]:
            furthest = definition
        if definition.kind == "array_bytes" and encoder is None:
            encoder = definition
    if encoder is None and not unclaimed:
        yield ValidationProblem(loc, "expected an array -> bytes codec, got none", "invalid_value")


def _chunk_problems(
    definition: CodecDefinition[Any],
    configuration: Mapping[str, Any],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> Problems:
    """What `definition`'s chunk rules find in a codec handed `chunk`, located under `at`."""
    return ruled(
        definition, lambda: definition.chunk_rules(read_only(configuration), nested, chunk), at
    )


def _inner_pipelines(
    definition: CodecDefinition[Any],
    configuration: Mapping[str, Any],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> tuple[Mapping[str, tuple[Stage, ...]], Problems]:
    """Each pipeline a codec handed `chunk` holds, read handed the chunk its `pipelines` says.

    Its codecs are the fields the member holding it holds, as the scope
    read them, and its problems are located there, under `at`. What
    `pipelines` gives is the extension author's code: it is checked to map
    members of the configuration to chunks, each member a list of codecs.
    """
    given = asked(
        definition,
        "pipelines",
        lambda: cast("object", definition.pipelines(read_only(configuration), nested, chunk)),
        at,
    )
    if not _is_pipelines(given):
        msg = (
            f"{definition.name!r}: its pipelines give a mapping of members to chunks, got {given!r}"
        )
        raise TypeError(msg)
    read: dict[str, tuple[Stage, ...]] = {}
    problems: list[ValidationProblem] = []
    for member, handed in given.items():
        codecs = _held(definition, configuration, nested, member)
        read[member], found = read_pipeline(codecs, handed, (*at, member))
        problems.extend(found)
    return read, tuple(problems)


def _is_pipelines(value: object) -> TypeGuard[Mapping[str, Chunk]]:
    """Whether `value` maps members of a configuration to chunks."""
    return is_object(value) and all(
        isinstance(member, str) and isinstance(chunk, Chunk) for member, chunk in value.items()
    )


def _held(
    definition: CodecDefinition[Any], configuration: Mapping[str, Any], nested: Nested, member: str
) -> tuple[Resolved[CodecDefinition[Any]], ...]:
    """The codecs `member` of the configuration holds, as the scope read them.

    A member holding anything but a list of fields read as codecs is a
    `TypeError`: a fault in the definition that names it.
    """
    entries: object = configuration.get(member)
    places = (
        [(member, index) for index in range(len(cast("tuple[object, ...]", entries)))]
        if isinstance(entries, tuple)
        else None
    )
    if places is None or not all(
        place in nested and nested[place].read_as is CodecDefinition for place in places
    ):
        msg = f"{definition.name!r}: its pipelines name {member!r}, which holds no list of codecs"
        raise TypeError(msg)
    return tuple(nested[place] for place in places)


def _handed_on(
    definition: CodecDefinition[Any],
    configuration: Mapping[str, Any],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> Chunk:
    """The chunk an array -> array codec hands on, handed `chunk`.

    Its `transition` is the extension author's code: what it gives is
    checked to be a chunk, and an error it raises says which codec's
    transition raised it, and where it was reading.
    """
    given = asked(
        definition,
        "transition",
        lambda: cast("object", definition.transition(read_only(configuration), nested, chunk)),
        at,
    )
    if not isinstance(given, Chunk):
        msg = f"{definition.name!r}: its transition gives a Chunk, got {given!r}"
        raise TypeError(msg)
    return given


__all__ = ["Stage", "read_pipeline"]
