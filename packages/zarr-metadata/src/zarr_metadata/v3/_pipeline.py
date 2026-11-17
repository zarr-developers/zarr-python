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

from zarr_metadata._json import (
    ValidationProblem,
    is_object,
    is_tuple,
    json_text,
    refine_user_data,
    with_input,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata.definition._definition import (
    AcceptedDeclaration,
    RefusedDeclaration,
    ResolvedDeclaration,
    UnclaimedDeclaration,
    asked,
    declaration_locations,
    in_document,
    item_at,
    own_key,
    read_only,
    respelled,
    ruled,
)
from zarr_metadata.definition._scope import claim_key, read_as_claimed, refines
from zarr_metadata.v3._definition import (
    ArrayArrayCodecDefinition,
    ArrayCodecDefinition,
    Chunk,
    CodecDefinition,
    CodecKind,
    unchanged_for_chunk,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from zarr_metadata._common import JSONValue
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

    codec: ResolvedDeclaration[CodecDefinition[Any]]
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
    codecs: Sequence[ResolvedDeclaration[CodecDefinition[Any]]], chunk: Chunk, loc: Loc = ()
) -> tuple[tuple[Stage, ...], Problems]:
    """Read `codecs` (codec declarations read by a context) as a pipeline that receives `chunk`. Return each codec with the chunk it receives, and the problems found.

    First the order is checked: array -> array codecs, then one
    array -> bytes codec, then bytes -> bytes codecs. Then each codec is
    processed in turn, with the chunk the previous codec passed on. Its
    chunk rules check it, and each problem is located where the document
    writes it: a member of its configuration in the configuration (or at
    the configuration itself, missing, for a codec that writes none), and
    a problem with the codec as a whole at the codec, as `_codec_problems`
    says. Each pipeline it contains is read the same way, at its location
    in the configuration. An array -> array codec is also asked what chunk it
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
        if not isinstance(definition, ArrayCodecDefinition):
            # A bytes -> bytes codec receives bytes, which are not a chunk.
            stages.append(Stage(codec, None))
            handed = None
            continue
        incoming = Chunk() if handed is None else handed
        at = (*loc, index, "configuration")
        inner = _no_stages()
        if isinstance(codec, AcceptedDeclaration):
            configuration, nested = codec.configuration, codec.nested
            found = _chunk_problems(definition, configuration, nested, incoming, at)
            problems.extend(_codec_problems(codec, (*loc, index), at, found))
            inner, found = _inner_pipelines(definition, configuration, nested, incoming, at)
            problems.extend(found)
        stages.append(Stage(codec, incoming, inner))
        if not isinstance(definition, ArrayArrayCodecDefinition):
            handed = None
        elif not isinstance(codec, AcceptedDeclaration):
            handed = Chunk()
        else:
            handed = _handed_on(definition, codec.configuration, codec.nested, incoming, at)
    if len(problems) == 0:
        return tuple(stages), ()
    return tuple(stages), with_input(problems, tuple(codec.json for codec in codecs), loc)


def _order_problems(
    codecs: Sequence[ResolvedDeclaration[CodecDefinition[Any]]], loc: Loc
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
    definition: ArrayCodecDefinition[Any],
    configuration: Mapping[str, object],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> Problems:
    """The problems `definition`'s chunk rules find in a codec that receives `chunk`, located under `at`."""
    return ruled(
        definition, lambda: definition.chunk_rules(read_only(configuration), nested, chunk), at
    )


def _codec_problems(
    codec: ResolvedDeclaration[CodecDefinition[Any]], here: Loc, at: Loc, found: Problems
) -> Problems:
    """`found`, the problems the chunk rules find in `codec` at `here`, located under `at`, its configuration, each moved to where the document writes it.

    A chunk rule judges the codec against the chunk it is handed, so a
    problem it reports at the configuration as a whole is the codec's,
    however the codec is spelled: `"bytes"` and a `bytes` with a
    configuration, handed values that vary in size, are one mistake at
    `here`. Any other is located as `in_document` locates a definition's
    rules.
    """
    return tuple(
        dataclasses.replace(each, loc=here)
        if each.loc == at
        else in_document(codec, here, (each,))[0]
        for each in found
    )


def _inner_pipelines(
    definition: ArrayCodecDefinition[Any],
    configuration: Mapping[str, object],
    nested: Nested,
    chunk: Chunk,
    at: Loc,
) -> tuple[Mapping[str, tuple[Stage, ...]], Problems]:
    """Read each pipeline contained in a codec that receives `chunk`, starting from the chunk that the codec's `pipelines` returns.

    A pipeline's codecs are the declarations in the configuration member that
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
) -> tuple[ResolvedDeclaration[CodecDefinition[Any]], ...]:
    """The codecs in configuration member `member`, as the scope read them.

    If the member holds anything other than a list of declarations read as
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
    definition: ArrayArrayCodecDefinition[Any],
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


def codec_for_chunk(stage: Stage) -> ResolvedDeclaration[CodecDefinition[Any]]:
    """The codec of `stage`, in the simplest spelling that means the same for the chunk it receives.

    A codec's own `canonical` reduces the spellings that mean the same
    thing wherever the codec is used. Some spellings mean the same thing
    only for some chunks. For example, a `bytes` codec with and without
    `endian` is one codec for values that take one byte, and two for
    values that take several. So a codec is compared, and written in its
    simplest spelling, as the codec this function returns: first each
    codec of each pipeline it contains, for the chunk that codec
    receives, and then its own configuration, as its definition's
    `chunk_canonical` simplifies it for the chunk in `stage`.
    `chunk_canonical` receives the configuration as the definition's
    `canonical` spells it, so two spellings that `canonical` reduces to
    one are simplified for the chunk in the same way.

    A codec the scope did not read is returned as it is. A codec's own
    configuration is kept as it is when the codec receives bytes, and
    when its chunk rules find a problem with it, because a codec with a
    problem has no simpler spelling. When nothing is simplified, the
    result is the codec itself. Otherwise it is a new declaration, read
    by the same definition. `canonical_of` writes it in its simplest
    spelling, and `==` compares it.

    `chunk_canonical` is the extension author's code, so its result is
    checked: the definition must allow it, the chunk rules must find no
    problem with it, it must hold the nested declarations the codec
    holds, each as it was, and `chunk_canonical` must have nothing left
    to simplify in it. A result that fails a check raises `ValueError`,
    which indicates a bug in the definition, not in the codec. A problem
    inside a nested declaration is the document's, and is not held
    against the result.
    """
    return _for_stage(stage.codec, stage)


def _for_stage(
    codec: ResolvedDeclaration[CodecDefinition[Any]], stage: Stage
) -> ResolvedDeclaration[CodecDefinition[Any]]:
    """`codec`, a reading of the codec at the place of `stage` in its pipeline, in the simplest spelling for the chunks that `stage` and the stages it contains receive.

    `codec` is the codec of `stage` itself, or the same codec as another
    scope read it. Each codec it holds is simplified for the chunk that
    the stage at its location receives.
    """
    if not isinstance(codec, AcceptedDeclaration):
        return codec
    inner: dict[Loc, ResolvedDeclaration[Any]] = {}
    for loc, each in _inner_stages(stage).items():
        held_there = codec.nested.get(loc)
        if held_there is None:
            continue
        simpler = _for_stage(held_there, each)
        if simpler is not held_there:
            inner[loc] = simpler
    configuration = _for_chunk(codec, stage.incoming)
    if configuration is None and len(inner) == 0:
        return codec
    return respelled(codec, codec.configuration if configuration is None else configuration, inner)


def _inner_stages(stage: Stage) -> dict[Loc, Stage]:
    """Each stage of the pipelines the codec of `stage` contains, by the location of its codec in the configuration."""
    return {
        (member, index): each
        for member, stages in stage.inner.items()
        for index, each in enumerate(stages)
    }


def codec_refines(stage: Stage, other: ResolvedDeclaration[CodecDefinition[Any]]) -> bool:
    """Whether the codec of `stage` refines `other`, another reading of the codec at that place in the pipeline.

    This is the order that `refines` defines for declarations, with each
    codec compared as `codec_for_chunk` compares it: in its simplest
    spelling for the chunk it receives. The chunk is the one `stage`
    knows. A scope that read less may know less of the chunk: when it
    claims no data type, or no codec that comes before, a `bytes` codec
    keeps an `endian` that `stage` knows to mean nothing. So `other` is
    simplified for the chunks of `stage` before the two are compared, and
    a document read in a richer scope refines the same document read in
    a poorer one.
    """
    mine = stage.codec
    if not isinstance(mine, AcceptedDeclaration) or isinstance(other, RefusedDeclaration):
        # Nothing is simplified for a codec the scope did not read.
        return refines(mine, other)
    if isinstance(other, UnclaimedDeclaration):
        # A gain, when what `other` reads as with the claims of `mine` is `mine`.
        if claim_key(mine) != claim_key(other):
            return False
        return _for_stage(read_as_claimed(other, mine), stage) == codec_for_chunk(stage)
    if mine.definition != other.definition or set(mine.nested) != set(other.nested):
        return False
    if own_key(_own_for_chunk(mine, stage.incoming)) != own_key(
        _own_for_chunk(other, stage.incoming)
    ):
        return False
    inner = _inner_stages(stage)
    return all(
        codec_refines(inner[loc], other.nested[loc])
        if loc in inner
        else refines(mine.nested[loc], other.nested[loc])
        for loc in mine.nested
    )


def _own_for_chunk(
    codec: AcceptedDeclaration[CodecDefinition[Any]], chunk: Chunk | None
) -> AcceptedDeclaration[CodecDefinition[Any]]:
    """`codec` with its own configuration as its `chunk_canonical` simplifies it for `chunk`, and each nested declaration as it is."""
    configuration = _for_chunk(codec, chunk)
    return codec if configuration is None else respelled(codec, configuration, {})


def _for_chunk(
    codec: AcceptedDeclaration[CodecDefinition[Any]], chunk: Chunk | None
) -> Mapping[str, JSONValue] | None:
    """The configuration of `codec` as its definition's `chunk_canonical` simplifies it for `chunk`, checked; None when nothing is simplified.

    `chunk_canonical` receives the configuration as the definition's
    `canonical` spells it. So every spelling that `canonical` reduces to
    one is simplified for the chunk in the same way, and `chunk_canonical`
    does not have to know each of them. None is returned when
    `chunk_canonical` changes nothing in that spelling, and the codec
    then stays as written, for `canonical` to simplify as it simplifies
    any configuration.
    """
    definition = codec.definition
    if chunk is None or not isinstance(definition, ArrayCodecDefinition):
        return None
    if definition.chunk_canonical is unchanged_for_chunk:
        return None
    configuration, nested = codec.configuration, codec.nested
    if len(_chunk_problems(definition, configuration, nested, chunk, ())) != 0:
        return None
    simplified = _asked_for_chunk(definition, configuration, nested, chunk)
    if simplified is None:
        return None
    _, refused = definition.read_configuration(simplified)
    held_at = frozenset(declaration_locations(definition, simplified))
    found = (
        # A problem inside a nested declaration is the document's: the
        # declaration is checked to be as it was written.
        *(problem for problem in refused if not _inside(problem.loc, nested)),
        *_chunk_problems(definition, simplified, nested, chunk, ()),
        *(
            ValidationProblem(
                loc, "expected the nested declaration to stay as it is", "invalid_value"
            )
            for loc in nested
            if _text_at(simplified, loc) != _text_at(cast("JSONValue", configuration), loc)
        ),
        *(
            ValidationProblem(
                loc, "expected no declaration that the codec did not hold", "invalid_value"
            )
            for loc in sorted(held_at.difference(nested), key=repr)
        ),
    )
    if len(found) != 0:
        msg = (
            f"{definition.name!r}: its chunk_canonical gave {simplified!r}, which does not hold: "
            f"{list(found)!r}"
        )
        raise ValueError(msg)
    # The simplest spelling has nothing left to simplify. Otherwise a
    # document written from it would read as another codec.
    further = _asked_for_chunk(definition, simplified, nested, chunk)
    if further is not None:
        msg = (
            f"{definition.name!r}: its chunk_canonical gave {simplified!r}, which is not the "
            f"simplest spelling: asked again, it gave {further!r}"
        )
        raise ValueError(msg)
    return simplified


def _inside(loc: Loc, nested: Nested) -> bool:
    """Whether `loc` is at or under the location of a declaration in `nested`."""
    return any(loc[: len(held_at)] == held_at for held_at in nested)


def _asked_for_chunk(
    definition: ArrayCodecDefinition[Any],
    configuration: Mapping[str, JSONValue],
    nested: Nested,
    chunk: Chunk,
) -> dict[str, JSONValue] | None:
    """What `definition`'s `chunk_canonical` gives for `configuration` as its `canonical` spells it, as JSON with arrays as tuples; None when it changes nothing in that spelling."""
    spelled = read_only(
        asked(
            definition,
            "canonical",
            lambda: dict(
                cast("Mapping[str, JSONValue]", definition.canonical(read_only(configuration)))
            ),
        )
    )
    given: object = asked(
        definition,
        "chunk_canonical",
        lambda: cast("object", definition.chunk_canonical(spelled, nested, chunk)),
    )
    if given is spelled:
        return None
    # The result is the extension author's value, so it is read as any JSON
    # is: an array may be a list, and what is not JSON is refused here,
    # naming the definition. `read_configuration` checks the rest.
    simplified, not_json = refine_user_data(dict(given)) if is_object(given) else (None, ())
    if not isinstance(simplified, dict) or len(not_json) != 0:
        msg = f"{definition.name!r}: its chunk_canonical gives a configuration, got {given!r}"
        raise TypeError(msg)
    if json_text(simplified) == json_text(cast("JSONValue", dict(spelled))):
        return None
    return simplified


def _text_at(configuration: JSONValue, loc: Loc) -> str | None:
    """The JSON text of the item of `configuration` at `loc`; None when `loc` does not lead into it."""
    item = item_at(configuration, loc)
    return None if item is UNSET else json_text(item)


__all__ = ["Stage", "codec_for_chunk", "codec_refines", "read_pipeline"]
