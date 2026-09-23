"""One metadata field, read against its definition.

Three steps, each feeding the next and needing more than the one before:

1. The type check needs the value and a TypedDict. `check` runs the
   checker compiled from the TypedDict's annotations over the JSON:
   its shape, member by member, every problem located, and a value of
   the TypedDict back -- a key it does not declare is reported and left
   out. A member holding another metadata field -- a shard's codecs, a
   cast's data type -- is annotated with a field alias, `CodecField`,
   and checked as an envelope: a name, or a named configuration. Which
   definition the name denotes is a scope's question, so the check
   needs none.
2. The rules need the definition: plain functions over the checked
   TypedDict, for everything finer than a type -- a bound, members read
   together. `Definition.judge` is the check and then the rules, for a
   caller holding one configuration.
3. The reading needs a scope. `resolve` relates the field's name to a
   definition through a `Context`, judges the configuration, and reads
   each nested field the check found, in the same scope.

A definition is a value, not a class to subclass. It holds the name the
metadata carries, the TypedDict that is the one declaration of the
configuration's JSON, and its rules, and it checks itself when it is
built. What kind of metadata it defines is its type -- `CodecDefinition`,
`ChunkGridDefinition` -- which is how a scope files it. Nothing happens
at class creation.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    TypeAlias,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import TypeAliasType, TypedDict, TypeVar, is_typeddict

from zarr_metadata._common import JSONValue, ZarrV3NamedConfigJSON
from zarr_metadata._json import ValidationProblem, refine_json
from zarr_metadata._typed_json import (
    Loc,
    Parsed,
    Parser,
    parser,
    problem,
    typeddict_keys,
    unread_in,
)
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON, validate_metadata_field_v3

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata.v3._registry import Context

C = TypeVar("C", default=Mapping[str, JSONValue])
"""A configuration: the TypedDict a definition declares its JSON as.

Any configuration is a `Mapping[str, JSONValue]`, which is what a kind
written without one stands for: `resolve(field, CodecDefinition, scope)`
reads as `CodecDefinition[Mapping[str, JSONValue]]`, nothing unknown.
"""

T = TypeVar("T")

D = TypeVar("D", bound="Definition[Any]")

Problems: TypeAlias = tuple[ValidationProblem, ...]


def no_rules(configuration: object) -> Iterator[ValidationProblem]:
    """The rules of a definition with none: every well-typed configuration is allowed."""
    yield from ()


def no_name_rules(name: str) -> Iterator[ValidationProblem]:
    """The name rules of a definition that claims one name: that name is allowed."""
    yield from ()


def unchanged(configuration: T) -> T:
    """The canonical form of a configuration with no simpler spelling: itself."""
    return configuration


class EmptyConfiguration(TypedDict, closed=True):
    """The configuration of a definition with nothing to configure, written as its bare name."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Definition(Generic[C]):
    """One extension's metadata, as JSON: its name, the TypedDict its configuration is, its rules.

    `configuration` is the TypedDict, and so the one declaration of the
    JSON: the checker is compiled from it, the static type of a checked
    configuration is it, and a document's author writes to it. It reads
    as the typing spec defines it -- `total`, `Required`, `NotRequired`,
    `closed` and `extra_items` mean what they mean to a type checker --
    and it says what a key it does not declare is: with `closed=True`, a
    problem, reported and left out; with `extra_items=`, a key of that
    type; with `closed=False`, anything. `rules` yields what the spec
    disallows in a configuration of that type, as it finds each; it is
    handed only a configuration that has passed the check, holding what
    the TypedDict admits and nothing else -- `judge` is the two, for a
    caller holding JSON.

    A family claims many names -- every `r<N>`, one data type -- through
    `names`, and says which of them are allowed through `name_rules`,
    whose problems land on the field, since a name is not configuration.
    Every other definition claims `name` alone. `canonical` is where two
    spellings of the configuration that mean the same thing are made one.

    Built by hand, a definition refuses what it could not read with: a
    `configuration` that is not a TypedDict, says nothing of the keys it
    does not declare, or has a member no checker reads, which is named;
    and a `name`, `names` or rules that are not what they say.
    """

    name: str
    """The name the metadata carries; for a family, the one it is filed under in a scope."""
    configuration: type[C]
    """The TypedDict the configuration is."""
    rules: Callable[[C], Iterable[ValidationProblem]] = no_rules
    """What the spec disallows in a well-typed configuration, located in it."""
    names: Callable[[str], bool] | None = None
    """Which names a family claims; None for a definition that claims `name` alone."""
    name_rules: Callable[[str], Iterable[ValidationProblem]] = no_name_rules
    """What the spec disallows in a claimed name."""
    canonical: Callable[[C], C] = unchanged
    """A well-typed, allowed configuration in its simplest equivalent spelling.

    Only the definition's own members: a nested field is put in its own
    canonical form by `canonicalize`, which knows where each one sits.
    """

    def __post_init__(self) -> None:
        refusal = _malformed(self) or self._refusal()
        if refusal is not None:
            raise TypeError(refusal)
        try:
            _vet(self.configuration)
        except TypeError as error:
            msg = f"{self.name!r}: {error}"
            raise TypeError(msg) from error

    def _refusal(self) -> str | None:
        """What is wrong with the members a kind adds; None when nothing is, or it adds none."""
        return None

    @property
    def requires_configuration(self) -> bool:
        """Whether a document must write a configuration: whether the TypedDict has a required key."""
        return len(typeddict_keys(self.configuration).required) != 0

    def claims(self, name: str) -> bool:
        """Whether `name` denotes this definition."""
        return name == self.name if self.names is None else self.names(name)

    def check(self, value: object, loc: Loc = ()) -> tuple[C | None, Problems]:
        """`value` type-checked as this definition's configuration, each nested field's envelope judged.

        `zarr_metadata.typed_json.check` is the type check alone; this also
        judges the envelope of each metadata field a member holds.
        """
        return _configuration_checked(value, self.configuration, loc)

    def judge(self, value: object, loc: Loc = ()) -> tuple[C | None, Problems]:
        """`value` type-checked, then judged by the rules: the configuration if it holds, and every problem.

        The rules are asked only of a configuration that type-checked and
        whose nested fields are well formed, holding what its TypedDict
        admits and nothing else, so a caller holding JSON never reaches a
        rule with a member of the wrong type, or one the type says cannot
        be there.
        """
        configuration, problems = self.check(value, loc)
        if configuration is None:
            return None, problems
        refused = _located(loc, tuple(self.rules(configuration)))
        return (configuration if len(refused) == 0 else None), (*problems, *refused)


def _malformed(definition: Definition[Any]) -> str | None:
    """What makes a hand-built definition unusable before its configuration is read; None if nothing does."""
    name = cast("object", definition.name)
    if not isinstance(name, str):
        return f"a definition's name is a string, got {name!r}"
    functions: dict[str, object] = {
        "rules": definition.rules,
        "name_rules": definition.name_rules,
        "canonical": definition.canonical,
    }
    if definition.names is not None:
        functions["names"] = definition.names
    return next(
        (
            f"{name!r}: {member} is a function, got {value!r}"
            for member, value in functions.items()
            if not callable(value)
        ),
        None,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class DataTypeDefinition(Definition[C]):
    """A data type."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ChunkGridDefinition(Definition[C]):
    """A chunk grid."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ChunkKeyEncodingDefinition(Definition[C]):
    """A chunk key encoding."""


CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""What a codec does to what it is handed: the three positions a pipeline orders."""


@dataclass(frozen=True, kw_only=True, slots=True)
class CodecDefinition(Definition[C]):
    """A codec, and what it does to what it is handed."""

    kind: CodecKind

    def _refusal(self) -> str | None:
        kind: object = self.kind
        if kind not in get_args(CodecKind):
            return f"{self.name!r}: kind is one of {get_args(CodecKind)!r}, got {kind!r}"
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class StorageTransformerDefinition(Definition[C]):
    """A storage transformer."""


KINDS: Final[tuple[type[Definition[Any]], ...]] = (
    DataTypeDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    StorageTransformerDefinition,
)
"""The kinds of metadata a document holds, each filed apart in a scope."""


def kind_of(definition: Definition[Any]) -> type[Definition[Any]] | None:
    """The kind `definition` is; None for a definition of no kind, which no scope files."""
    return next((kind for kind in KINDS if isinstance(definition, kind)), None)


def as_kind(kind: object) -> type[Definition[Any]]:
    """The kind of metadata `kind` names, type arguments dropped; `TypeError` if it names none.

    A scope files definitions by kind, so a field is read as one of
    `KINDS` -- `CodecDefinition`, or `CodecDefinition[Any]` -- and never
    as the base `Definition` or a class of the caller's own, under which
    nothing is filed: a field read as one would go unjudged.
    """
    origin = get_origin(kind) or kind
    found = next((known for known in KINDS if origin is known), None)
    if found is None:
        names = ", ".join(known.__name__ for known in KINDS)
        msg = f"{kind!r} is not a kind of metadata; read a field as one of {names}"
        raise TypeError(msg)
    return found


DataTypeField = TypeAliasType("DataTypeField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a data type, read in the scope the member's field is read in."""
ChunkGridField = TypeAliasType("ChunkGridField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a chunk grid."""
ChunkKeyEncodingField = TypeAliasType("ChunkKeyEncodingField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a chunk key encoding."""
CodecField = TypeAliasType("CodecField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a codec: a shard's `codecs` is `tuple[CodecField, ...]`."""
StorageTransformerField = TypeAliasType("StorageTransformerField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a storage transformer."""

_FIELD_KINDS: Final[Mapping[object, type[Definition[Any]]]] = {
    DataTypeField: DataTypeDefinition,
    ChunkGridField: ChunkGridDefinition,
    ChunkKeyEncodingField: ChunkKeyEncodingDefinition,
    CodecField: CodecDefinition,
    StorageTransformerField: StorageTransformerDefinition,
}


@dataclass(frozen=True, slots=True)
class _NestedField:
    """A metadata field the check met inside a configuration: where it sits, its kind, its JSON.

    What the checker hands back where a field alias is, in place of the
    JSON: `_checked` collects each one and puts its JSON back. Carried in
    the value rather than recorded on the side, so a branch of a union
    that did not match leaves none behind.
    """

    loc: Loc
    kind: type[Definition[Any]]
    json: JSONValue


def _field(annotation: object) -> Parser | None:
    """The checker's one shape of this module's own: a member holding a metadata field.

    Checked as a bare name or an object, and handed on: the envelope is
    judged, and the name related to a definition, by whoever reads the
    field -- `check` without a scope, `resolve` in one.
    """
    try:
        kind = _FIELD_KINDS.get(annotation)
    except TypeError:  # an unhashable annotation is no field alias
        return None
    if kind is None:
        return None

    def parse(value: object, loc: Loc) -> Parsed:
        if not isinstance(value, (str, Mapping)):
            return value, problem(loc, f"expected a metadata field, got {value!r}")
        # Refined JSON, which the checker only knows as `object`.
        return _NestedField(loc, kind, cast("JSONValue", value)), ()

    return parse


def _vetting(annotation: object) -> Parser | None:
    """`_field`, refusing what a definition's configuration cannot hold and still be read.

    A member typed with the envelope's own TypedDict --
    `ZarrV3MetadataFieldJSON` -- checks as plain JSON: its name would never
    be related to a definition, nor its configuration judged. And a
    TypedDict that says nothing of the keys it does not declare is open,
    so a key it does not declare would never be reported.
    """
    if annotation is ZarrV3NamedConfigJSON:
        msg = (
            "a member typed ZarrV3MetadataFieldJSON checks as plain JSON, and is never read as "
            "a field; annotate it with the field alias of its kind: "
            + ", ".join(cast("TypeAliasType", alias).__name__ for alias in _FIELD_KINDS)
        )
        raise TypeError(msg)
    if (
        isinstance(annotation, type)
        and is_typeddict(annotation)
        and not typeddict_keys(annotation).declared
    ):
        msg = (
            f"{annotation.__name__} says nothing of the keys it does not declare, so it is "
            "open, and such a key would go unreported; declare it closed=True, or "
            "extra_items= for what such a key holds, or closed=False to take any"
        )
        raise TypeError(msg)
    return _field(annotation)


@functools.cache
def _vet(configuration: type) -> None:
    """Refuse a configuration no definition could read with, saying what is wrong with it."""
    if not is_typeddict(configuration):
        msg = (
            f"configuration is {configuration!r}; give the TypedDict the configuration's JSON takes"
        )
        raise TypeError(msg)
    _vetting(configuration)
    unread = unread_in(configuration, _vetting)
    if unread is not None:
        msg = f"{unread}; a finer rule goes in the definition's `rules`"
        raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class _Checker:
    """A TypedDict's checker, compiled once, and whether a value of it can hold a nested field."""

    parse: Parser
    nests: bool


@functools.cache
def _checker(shape: type) -> _Checker:
    """The checker for a TypedDict, compiled once; `TypeError` naming what no checker reads."""
    met: list[object] = []

    def leaf(annotation: object) -> Parser | None:
        found = _field(annotation)
        if found is not None:
            met.append(annotation)
        return found

    return _Checker(parser(shape, leaf), len(met) != 0)


def _checked(
    shape: type, value: object, loc: Loc
) -> tuple[object, Problems, tuple[_NestedField, ...]]:
    """`value` checked as `shape`: the typed value, every problem, and the fields nested in it, in order."""
    checker = _checker(shape)
    typed, found = checker.parse(value, loc)
    if not checker.nests:
        return typed, found, ()
    nested: list[_NestedField] = []
    return _put_back(typed, nested), found, tuple(nested)


def _put_back(value: object, nested: list[_NestedField]) -> object:
    """`value` with each nested field the checker handed back put back as its JSON, collected in order."""
    if isinstance(value, _NestedField):
        nested.append(value)
        return value.json
    if isinstance(value, tuple):
        return tuple(_put_back(entry, nested) for entry in cast("tuple[object, ...]", value))
    if isinstance(value, dict):
        entries = cast("dict[str, object]", value)
        return {key: _put_back(entry, nested) for key, entry in entries.items()}
    return value


def _usable(problems: Sequence[ValidationProblem]) -> bool:
    """Whether a value with these problems still reads: an unknown key is survivable, nothing else is."""
    return all(found.kind == "unknown_key" for found in problems)


def _located(prefix: Loc, problems: Iterable[ValidationProblem]) -> Problems:
    return tuple(
        ValidationProblem((*prefix, *found.loc), found.message, found.kind) for found in problems
    )


def _envelope(field: _NestedField) -> Problems:
    """What is wrong with a nested field's envelope, at the field."""
    return _located(
        field.loc, validate_metadata_field_v3(field.json, allow_must_understand_false=False)
    )


def _configuration_checked(
    value: object, shape: type[T], loc: Loc = ()
) -> tuple[T | None, Problems]:
    """`value` checked as `shape`, as `typed_json.check` checks it, and each nested field's envelope judged.

    The step a definition's `judge` starts from. A member typed with a
    field alias holds a metadata field, whose envelope is judged as a
    document's is: a stray member, or a `must_understand` of `false`, is a
    problem of the configuration, and the value does not come back.
    """
    if not is_typeddict(shape):
        msg = f"{shape!r} is not a TypedDict"
        raise TypeError(msg)
    refined, problems = refine_json(value, loc)
    if refined is None:
        return None, problems
    typed, found, nested = _checked(shape, refined, loc)
    problems = (*found, *(problem for field in nested for problem in _envelope(field)))
    return (cast("T", typed) if _usable(problems) else None), problems


def named_configuration(
    value: object,
) -> tuple[str | None, Mapping[str, object] | None, Problems]:
    """Split a metadata field into `(name, configuration, problems)`.

    A bare name, or an object carrying one. A `None` name means the value
    is not a metadata field at all; a `None` configuration means the bare
    spelling was used, or the key was left out. A configuration that is
    present and not an object is the one problem reported, at
    `("configuration",)`.
    """
    if isinstance(value, str):
        return value, None, ()
    if not isinstance(value, Mapping):
        return None, None, ()
    entry = cast("Mapping[str, object]", value)
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, ()
    if "configuration" not in entry:
        return name, None, ()
    configuration = entry["configuration"]
    if not isinstance(configuration, Mapping):
        return name, None, problem(("configuration",), f"expected an object, got {configuration!r}")
    return name, cast("Mapping[str, object]", configuration), ()


Unread = Literal["out_of_scope", "invalid"]
"""A field no definition read: nothing in scope claims its name, or it could not be read."""

Resolution = Literal["read"] | Unread
"""What a scope made of a field: read by the definition that claims it, or unread, and why."""


@dataclass(frozen=True, slots=True)
class Resolved(Generic[D]):
    """One metadata field, as read in a scope: its JSON, and what the scope made of it.

    `resolution` is what became of the configuration: read by the
    definition that claims the name, claimed by nothing, or not readable.
    A problem with the envelope around it -- a stray member, a
    `must_understand` of `false` -- is reported with the field, and leaves
    the resolution as it is.
    """

    json: JSONValue
    """The field as written, refined: arrays as tuples. `None` for a value that was not JSON."""
    resolution: Resolution
    definition: D | None
    """The definition that claims the field's name; None when nothing in scope does, or it names none."""
    configuration: Mapping[str, JSONValue] | None
    """The configuration, type-checked and allowed by the rules, when the field was read; None otherwise."""


def configuration_of(resolved: Resolved[Any], definition: Definition[C]) -> C | None:
    """The configuration `resolved` holds, typed as `definition` declares it, if `definition` read it.

    `Resolved` holds a configuration as the mapping every one is; asked
    with the definition that read the field, this is the same mapping, as
    its TypedDict. None when another definition read it, or none did.
    """
    if resolved.definition is not definition or resolved.configuration is None:
        return None
    return cast("C", resolved.configuration)


def resolve(
    data: object, kind: type[D], context: Context, loc: Loc = ()
) -> tuple[Resolved[D], Problems]:
    """`data`, one metadata field, read as a `kind` in `context`: what the scope made of it, and every problem.

    All three steps for one field. `data` is refined to JSON and its
    envelope judged -- an extra member, a `configuration` that is not an
    object, a `must_understand` that is not a boolean or is `false`, each
    a problem. The name is related to a definition in `context`; the
    configuration is checked against its TypedDict and judged by its
    rules; each nested field the check met is read the same way, in the
    same scope. A name nothing claims is `out_of_scope`: an unmodelled
    extension, left unjudged, which is what keeps the format open. `loc`
    prefixes every problem. `kind` is one of `KINDS`, with or without
    type arguments; anything else is a `TypeError`.
    """
    asked = as_kind(kind)
    refined, problems = refine_json(data, loc)
    if refined is None:
        return Resolved(None, "invalid", None, None), problems
    resolved, found = _resolve_field(refined, asked, context, loc)
    return cast("Resolved[D]", resolved), found


def _resolve_field(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[Resolved[Definition[Any]], Problems]:
    """A refined field with its envelope judged, then read.

    The resolution is what became of the configuration. A stray member or
    a `must_understand` of `false` says nothing about it, so it is reported
    beside the field that was read, which later layers can still judge.
    """
    envelope = _located(loc, validate_metadata_field_v3(data, allow_must_understand_false=False))
    resolved, found = _read(data, kind, context, loc)
    return resolved, (*envelope, *found)


def _read(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[Resolved[Definition[Any]], Problems]:
    name, given, malformed = named_configuration(data)
    if name is None or len(malformed) != 0:
        return Resolved(data, "invalid", None, None), ()
    definition = context.claimant(kind, name)
    if definition is None:
        return Resolved(data, "out_of_scope", None, None), ()
    at = (*loc, "configuration")
    problems = list(_located(loc, definition.name_rules(name)))
    if given is None and definition.requires_configuration:
        problems.extend(problem(at, f"{name!r} requires a configuration", "missing_key"))
        return Resolved(data, "invalid", definition, None), tuple(problems)
    typed, found, nested = _checked(definition.configuration, {} if given is None else given, at)
    problems.extend(found)
    envelopes = [_envelope(field) for field in nested]
    sound = _usable(found) and all(_usable(envelope) for envelope in envelopes)
    configuration = cast("Mapping[str, JSONValue]", typed) if sound else None
    if configuration is not None:
        problems.extend(_located(at, definition.rules(configuration)))
    for field, envelope in zip(nested, envelopes, strict=True):
        problems.extend(envelope)
        problems.extend(_read(field.json, field.kind, context, field.loc)[1])
    if not _usable(problems):
        return Resolved(data, "invalid", definition, None), tuple(problems)
    return Resolved(data, "read", definition, configuration), tuple(problems)


def canonicalize(
    data: object, kind: type[D], context: Context, loc: Loc = ()
) -> tuple[JSONValue | None, Problems]:
    """`data`, one metadata field, in its simplest equivalent spelling, and every problem.

    Only a field that reads has one. Its configuration holds what its
    TypedDict admits, each nested field goes in its own canonical form,
    and then the definition's `canonical` has the rest -- judged again,
    so a `canonical` that gives a configuration that does not hold is a
    `ValueError`, a fault in the definition rather than the field. The
    envelope takes the fewest words: the bare name when nothing is
    configured, and no `must_understand`, since `true` is what absence
    means and `false` is refused, a problem reported with the field. A
    name nothing in scope claims comes back as written, since what it
    simplifies to is its own definition's call; a field that does not
    read has no canonical form, and comes back None.
    """
    resolved, problems = resolve(data, kind, context, loc)
    if resolved.resolution == "out_of_scope":
        return resolved.json, problems
    if resolved.resolution != "read" or resolved.definition is None:
        return None, problems
    return _canonical_field(resolved.definition, resolved, context), problems


def _canonical_field(
    definition: Definition[Any], resolved: Resolved[Any], context: Context
) -> JSONValue:
    """A field that read, in its simplest equivalent spelling: nested fields first, then its own members."""
    name, _, _ = named_configuration(resolved.json)
    configuration: JSONValue = dict(resolved.configuration or {})
    _, _, nested = _checked(definition.configuration, configuration, ())
    for field in nested:
        simplest, _ = canonicalize(field.json, field.kind, context)
        configuration = _replaced(
            configuration, field.loc, field.json if simplest is None else simplest
        )
    simplified = cast("Mapping[str, JSONValue]", definition.canonical(configuration))
    _, refused = definition.judge(simplified)
    if len(refused) != 0:
        msg = (
            f"{definition.name!r}: its canonical gave {simplified!r}, which does not hold: "
            f"{list(refused)!r}"
        )
        raise ValueError(msg)
    if len(simplified) == 0:
        return name
    return {"name": name, "configuration": simplified}


def _replaced(value: JSONValue, path: Loc, new: JSONValue) -> JSONValue:
    """`value` with what sits at `path` replaced by `new`; `path` comes from a check of `value`."""
    if len(path) == 0:
        return new
    step, rest = path[0], path[1:]
    if isinstance(step, str):
        members = cast("Mapping[str, JSONValue]", value)
        return {**members, step: _replaced(members[step], rest, new)}
    entries = cast("tuple[JSONValue, ...]", value)
    return (*entries[:step], _replaced(entries[step], rest, new), *entries[step + 1 :])


__all__ = [
    "KINDS",
    "ChunkGridDefinition",
    "ChunkGridField",
    "ChunkKeyEncodingDefinition",
    "ChunkKeyEncodingField",
    "CodecDefinition",
    "CodecField",
    "CodecKind",
    "DataTypeDefinition",
    "DataTypeField",
    "Definition",
    "EmptyConfiguration",
    "Resolution",
    "Resolved",
    "StorageTransformerDefinition",
    "StorageTransformerField",
    "Unread",
    "as_kind",
    "canonicalize",
    "configuration_of",
    "kind_of",
    "named_configuration",
    "no_name_rules",
    "no_rules",
    "resolve",
    "unchanged",
]
