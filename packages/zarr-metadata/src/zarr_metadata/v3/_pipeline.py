"""Reading a codec pipeline: each codec paired with the chunk it receives.

A pipeline is array -> array codecs, then one array -> bytes codec, then
bytes -> bytes codecs. The array passes a `Chunk` to its first codec. The
chunk has "the same data type as the Zarr array, and shape equal to the
chunk shape"
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1048-L1050),
where the shape is the lengths of the grid's chunks along each axis. Each
array -> array codec passes the next codec the chunk that its `transition`
returns.
Each codec that receives an array is checked by its chunk rules against
that chunk. For example, a `bytes` codec that receives a multi-byte data
type without an `endian` is a problem, and so is a `transpose` whose
`order` has the wrong number of axes. If a codec contains its own
pipelines, each one is read the same way, starting from the chunk that the
codec's `pipelines` returns. For a shard, the inner codecs receive the
inner chunks, and the index codecs receive the shard index.

The reader does not guess. If a codec was not read (no definition in the
scope claims its name, or its configuration has problems), its output is
unknown, so the next codec receives a chunk with unknown shape and data
type. The same applies after a codec that does not declare its output. A
later codec can still declare its output itself, as `cast_value` declares
its data type. An unclaimed codec could be any of the three kinds, so the
order check skips it. A chunk rule checks only the properties of the
chunk that are known.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeGuard, cast

from zarr_metadata._json import ValidationProblem, is_object, is_tuple, with_input
from zarr_metadata.definition._definition import (
    AcceptedField,
    ResolvedField,
    asked,
    read_only,
    ruled,
)
from zarr_metadata.v3._definition import (
    Chunk,
    CodecDefinition,
    CodecKind,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import (
        Nested,
        Problems,
    )


def _no_stages() -> Mapping[str, tuple[Stage, ...]]:
    """The pipelines of a codec that contains none, or that was not read: an empty mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class Stage:
    """One codec of a pipeline, and the chunk it receives."""

    codec: ResolvedField[CodecDefinition[Any]]
    """The codec, as the scope read it."""
    incoming: Chunk | None
    """The chunk the codec receives.

    None for a codec that receives bytes (a bytes -> bytes codec, or any
    codec after the array -> bytes codec). Also None for an unclaimed codec
    when its input is not known to be an array.
    """
    inner: Mapping[str, tuple[Stage, ...]] = dataclasses.field(default_factory=_no_stages)
    """The pipelines the codec contains, keyed by the configuration member that holds each one. Each pipeline pairs each codec with the chunk it receives."""

    def __post_init__(self) -> None:
        # Make this read-only, like everything else a reading returns.
        object.__setattr__(self, "inner", MappingProxyType(dict(self.inner)))

    def __reduce__(self) -> tuple[type[Stage], tuple[object, ...]]:
        # A read-only view cannot be pickled, so pickle the stage from its constructor arguments.
        return Stage, (self.codec, self.incoming, dict(self.inner))


_POSITIONS: Final[Mapping[CodecKind, int]] = {
    "array_array": 0,
    "array_bytes": 1,
    "bytes_bytes": 2,
}
"""The position of each kind of codec in a pipeline."""

_SPOKEN: Final[Mapping[CodecKind, str]] = {
    "array_array": "array -> array",
    "array_bytes": "array -> bytes",
    "bytes_bytes": "bytes -> bytes",
}


def read_pipeline(
    codecs: Sequence[ResolvedField[CodecDefinition[Any]]], chunk: Chunk, loc: Loc = ()
) -> tuple[tuple[Stage, ...], Problems]:
    """Read `codecs` (codec fields read by a context) as a pipeline that receives `chunk`. Return each codec with the chunk it receives, and the problems found.

    First the order is checked: array -> array codecs, then one
    array -> bytes codec, then bytes -> bytes codecs. Then each codec is
    processed in turn, with the chunk the previous codec passed on. Its
    chunk rules check it and locate their problems in its configuration.
    Each pipeline it contains is read the same way, at its location in the
    configuration. An array -> array codec is also asked what chunk it
    passes on. Problems are located under `loc`, the location of the
    codecs, with each codec at its index.
    """
    problems = list(_order_problems(codecs, loc))
    stages: list[Stage] = []
    # The chunk the next codec receives. None once the input is not known
    # to be an array: after the array -> bytes codec, or after a codec of
    # unknown kind.
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
        if isinstance(codec, AcceptedField):
            configuration, nested = codec.configuration, codec.nested
            problems.extend(_chunk_problems(definition, configuration, nested, incoming, at))
            inner, found = _inner_pipelines(definition, configuration, nested, incoming, at)
            problems.extend(found)
        stages.append(Stage(codec, incoming, inner))
        if definition.kind == "array_bytes":
            handed = None
        elif not isinstance(codec, AcceptedField):
            handed = Chunk()
        else:
            handed = _handed_on(definition, codec.configuration, codec.nested, incoming, at)
    if len(problems) == 0:
        return tuple(stages), ()
    return tuple(stages), with_input(problems, tuple(codec.json for codec in codecs), loc)


def _order_problems(
    codecs: Sequence[ResolvedField[CodecDefinition[Any]]], loc: Loc
) -> Iterator[ValidationProblem]:
    """Array -> array codecs, then one array -> bytes codec, then bytes -> bytes codecs.

    "the list of codecs must be of the following form: zero or more
    array -> array codecs; followed by exactly one array -> bytes codec;
    followed by zero or more bytes -> bytes codecs"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L974-L983).
    An unclaimed codec is skipped. It could be any of the three kinds, so a
    pipeline that contains one is not rejected for lacking an
    array -> bytes codec. A codec in the wrong position is reported at its
    location, and so is any array -> bytes codec after the first.
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
    configuration: Mapping[str, object],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> Problems:
    """The problems `definition`'s chunk rules find in a codec that receives `chunk`, located under `at`."""
    return ruled(
        definition, lambda: definition.chunk_rules(read_only(configuration), nested, chunk), at
    )


def _inner_pipelines(
    definition: CodecDefinition[Any],
    configuration: Mapping[str, object],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> tuple[Mapping[str, tuple[Stage, ...]], Problems]:
    """Read each pipeline contained in a codec that receives `chunk`, starting from the chunk that the codec's `pipelines` returns.

    A pipeline's codecs are the fields in the configuration member that
    holds it, as the scope read them. Its problems are located at that
    member, under `at`. `pipelines` is the extension author's code, so its
    result is checked: it must map configuration members to chunks, and
    each member must be a list of codecs.
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
    definition: CodecDefinition[Any],
    configuration: Mapping[str, object],
    nested: Nested,
    member: str,
) -> tuple[ResolvedField[CodecDefinition[Any]], ...]:
    """The codecs in configuration member `member`, as the scope read them.

    If the member holds anything other than a list of fields read as
    codecs, this raises `TypeError`, because the definition that names the
    member has a bug.
    """
    entries: object = configuration.get(member)
    places = [(member, index) for index in range(len(entries))] if is_tuple(entries) else None
    if places is None or not all(
        place in nested and nested[place].read_as is CodecDefinition for place in places
    ):
        msg = f"{definition.name!r}: its pipelines name {member!r}, which holds no list of codecs"
        raise TypeError(msg)
    return tuple(nested[place] for place in places)


def _handed_on(
    definition: CodecDefinition[Any],
    configuration: Mapping[str, object],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> Chunk:
    """The chunk an array -> array codec passes on when it receives `chunk`.

    The codec's `transition` is the extension author's code. Its result is
    checked to be a chunk. If it raises an error, the error says which
    codec's transition raised it and where the reader was reading.
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
