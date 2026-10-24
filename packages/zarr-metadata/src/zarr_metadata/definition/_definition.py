"""Reading one metadata field against its definition.

There are three steps. Each step feeds the next and needs more input than
the step before:

1. The type check needs the value and a TypedDict. `check` runs the
   checker compiled from the TypedDict's annotations over the JSON. It
   checks the shape member by member, locates every problem, and returns
   a value of the TypedDict. A key the TypedDict does not declare is
   reported and left out. A member that holds another metadata field (a
   shard's codecs, a cast's data type) is annotated with a field alias
   such as `ZarrV3CodecField`, and is checked as an envelope: a name, or
   a named configuration. Which definition the name refers to is a
   question for a scope, so the check does not need one.
2. The rules need the definition. They are plain functions over the
   checked TypedDict, for constraints that a type cannot express (a
   bound, or members that depend on each other).
   `Definition.read_configuration` runs the check and then the rules, for
   a caller that has one configuration.
3. The reading needs a scope. `resolve` maps the field's name to a
   definition through a `Context`, checks the configuration, and reads
   each nested field found by the check, in the same scope.

A definition is a value, not a class to subclass. It holds the name the
metadata uses, the TypedDict that is the single declaration of the
configuration's JSON, and its rules. It checks itself when it is built.
The kind of metadata it defines is its type (`CodecDefinition`,
`ChunkGridDefinition`), and a scope files it by that type. Building a
definition files it nowhere: only a scope built from it does.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    cast,
    get_origin,
    get_type_hints,
)

from typing_extensions import TypeAliasType, TypedDict, TypeVar, is_typeddict

from zarr_metadata._common import JSONValue, ZarrV3NamedConfigJSON
from zarr_metadata._json import (
    Problems,
    ValidationProblem,
    copied,
    frozen,
    is_object,
    is_tuple,
    json_text,
    refine_json,
    refine_user_data,
    shown,
    with_input,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata._typed_json import (
    Loc,
    Parsed,
    Parser,
    is_alias,
    parser,
    problem,
    typeddict_keys,
    unread_in,
)
from zarr_metadata.definition._envelope import ENVELOPE_KEYS, envelope_problems, name_problem

if TYPE_CHECKING:
    from collections.abc import Sequence

C = TypeVar("C", default=Mapping[str, JSONValue])
"""A configuration: the TypedDict that a definition uses to declare its JSON.

Any configuration is a `Mapping[str, JSONValue]`. That is the default when
a kind is written without a type argument:
`resolve(field, CodecDefinition, context)` is typed as
`CodecDefinition[Mapping[str, JSONValue]]`, with no unknown types.
"""


T = TypeVar("T")


D = TypeVar("D", bound="Definition[Any]")


F = TypeVar("F", bound="WithFillValue[Any]")


def no_rules(*_: object) -> Iterator[ValidationProblem]:
    """The rules of a definition (of any kind) that has no rules. Every well-typed configuration is allowed."""
    yield from ()


def unchanged(configuration: T) -> T:
    """The canonical form of a configuration that has no simpler spelling: the configuration itself."""
    return configuration


def fill_value_as_written(configuration: object, nested: object, value: T) -> T:
    """The canonical spelling of a fill value for a data type that has only one spelling per value: the fill value as written."""
    return value


class EmptyConfiguration(TypedDict, closed=True):
    """The configuration of a definition with nothing to configure. Its field is written with its name only."""


_FIELD_KINDS: Final[dict[TypeAliasType, type[Definition[Any]]]] = {}
"""Maps each field alias to the kind that a member annotated with it is read as. Each kind adds its aliases when its class is built."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Definition(Generic[C]):
    """One extension's metadata as JSON: its name, its configuration TypedDict, and its rules.

    `configuration` is the TypedDict, and so it is the single declaration
    of the JSON. The checker is compiled from it, it is the static type of
    a checked configuration, and a document's author writes to it. It is
    interpreted as the typing spec defines a TypedDict: `total`,
    `Required`, `NotRequired`, `closed` and `extra_items` mean what they
    mean to a type checker. It also declares how keys it does not declare
    are handled. With `closed=True`, such a key is a problem, reported and
    left out. With `extra_items=`, such a key must have that type. With
    `closed=False`, anything is allowed. `rules` yields each violation of
    the spec it finds in a configuration of that type. It receives only a
    configuration that has passed the check: it contains only what the
    TypedDict admits, each member is within the bounds its type declares,
    and the nested fields are passed as the scope read them (for example, a
    struct's field types). The nested fields are empty when no scope read
    the configuration. `read_configuration` runs both steps, for a caller
    that has JSON.

    Each function receives the configuration as a read-only view, not a
    `dict`. `copy.deepcopy` and `json.dumps` reject this view. A function
    that simplifies a spelling must build a new mapping, such as
    `{**configuration}` without the member, instead of editing its input.

    `canonical` is where two spellings of the configuration that mean the
    same thing are reduced to one.

    A definition built by hand rejects inputs it could not read with: a
    `configuration` that is not a TypedDict, that does not declare how
    undeclared keys are handled, or that has a member no checker reads
    (the error names the member); and a `name` or rules of the wrong type.
    """

    _is_kind: ClassVar[bool] = False
    """Whether this class is a kind, which is what a scope files definitions by. Set by `class Kind(Definition, kind=True, format=3)`.

    Read from the class's own namespace and never inherited: a subclass of
    a kind is a definition of that kind, and does not declare `kind`.
    """
    _format: ClassVar[Literal[2, 3] | None] = None
    """The Zarr format of the documents that contain fields of this kind. It is declared with the kind and inherited by its definitions, and it sets the format a scope reads."""
    label: ClassVar[str] = "definition"
    """The name of the kind in error messages, for example "codec"."""
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = ()
    """The field aliases used to annotate a configuration member that holds a field of this kind. For a codec: `ZarrV3CodecField` and `ZarrV3StaticCodecField`."""

    name: str
    """The name the metadata uses. A scope files the definition under this name."""
    configuration: type[C]
    """The configuration TypedDict."""
    rules: Callable[[C, Nested], Iterable[ValidationProblem]] = no_rules
    """The spec violations in a well-typed configuration and its nested fields, located in the configuration."""
    canonical: Callable[[C], C] = unchanged
    """A well-typed, allowed configuration in its simplest equivalent spelling.

    This handles only the definition's own members. `canonicalize` puts
    each nested field in its own canonical form, because it knows the
    location of each one.
    """

    @classmethod
    def name_problem(cls, name: str, at: Loc) -> ValidationProblem | None:
        """The problem to report at `at` when no document of the format may use `name` for a field of this kind; None when it may. For Zarr v3, a name may be used when it is a valid extension name under the spec."""
        return name_problem(name, at)

    @classmethod
    def well_named(cls, name: str) -> bool:
        """Whether a document of the format may use `name` for a field of this kind, as decided by `name_problem`."""
        return cls.name_problem(name, ()) is None

    @classmethod
    def filed_under(cls, name: str) -> tuple[str | None, dict[str, JSONValue] | None]:
        """Interpret a name used in a document: return the name its definition is filed under, and the configuration encoded in the name.

        By default a name is filed as itself and encodes nothing:
        `(name, None)`. A kind whose names encode configuration overrides
        this. For example, the v3 data type `r16` is filed under `r*` with
        `{"bits": 16}`. A name that no document uses, and that exists only
        to file a definition, returns `(None, None)`.
        """
        return name, None

    def carrying_name(self, configuration: Mapping[str, JSONValue]) -> str | None:
        """The name that encodes `configuration` for this definition. This is the inverse of `filed_under`: `r16` for `r*` with `{"bits": 16}`. None when this definition's names encode nothing."""
        return None

    @classmethod
    def named_configuration(
        cls, value: object
    ) -> tuple[str | None, Mapping[str, object] | None, Problems]:
        """Split `value` (a field as a document of the format writes it) into `(name, configuration, problems)`, as the module-level `named_configuration` does for a v3 field."""
        return named_configuration(value)

    @classmethod
    def envelope_problems(cls, value: object) -> Problems:
        """Every reason `value` is not a valid envelope for a field of this kind in this format. The contents of the configuration are not checked."""
        return envelope_problems(value, allow_must_understand_false=False)

    @classmethod
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        """A field with `name` and `configuration`, as a document of the format writes it, in the shortest form every reader accepts. For v3 this is an object."""
        if len(configuration) != 0:
            return {"name": name, "configuration": configuration}
        return {"name": name}

    @classmethod
    def unclaimed_json(
        cls, json: JSONValue, name: str, configuration: Mapping[str, JSONValue]
    ) -> JSONValue:
        """A field of this kind that no definition claims, as a document writes it: `json` is the field as written. By default the name and configuration go in the envelope every reader accepts, as for a field a definition read."""
        return cls.envelope_json(name, configuration)

    @classmethod
    def refined(
        cls, data: object, loc: Loc, definition: Definition[Any] | None
    ) -> tuple[JSONValue | None, Problems]:
        """`data`, a field of this kind at `loc`, refined to JSON before it is read, with every reason it is not JSON. `definition` is the definition that claims the field, if any. By default every number must be finite, as RFC 8259 requires."""
        return refine_json(data, loc)

    @classmethod
    def configuration_loc(cls, loc: Loc) -> Loc:
        """The location of the configuration of a field at `loc`: under `configuration` for v3, or at the field itself for a format that writes the parameters next to the name."""
        return (*loc, "configuration")

    @classmethod
    def name_loc(cls, loc: Loc) -> Loc:
        """The location of the name of a field at `loc` that is written as an object: under `name` for v3."""
        return (*loc, "name")

    @classmethod
    def alias_problems(
        cls, alias: TypeAliasType, definition: Definition[Any], loc: Loc
    ) -> Problems:
        """The problems with using `definition` in a member annotated with `alias`, one of this kind's `field_aliases`. Empty by default.

        A kind with an alias that admits only some of its definitions
        overrides this. For example, `ZarrV3StaticCodecField` admits only
        codecs of static size.
        """
        return ()

    def __init_subclass__(
        cls, *, kind: bool = False, format: Literal[2, 3] | None = None, **kwargs: object
    ) -> None:
        """Register a subclass. `kind=True` declares a kind, with the Zarr `format` of the documents that contain its fields. This marks the class but not its subclasses, as `typing.Protocol` and SQLAlchemy's `__abstract__` do."""
        # Use the explicit form, not bare `super()`. A dataclass with slots
        # is rebuilt, and the cell that bare `super()` reads refers to the
        # discarded class.
        super(Definition, cls).__init_subclass__(**kwargs)
        # A dataclass with slots is built twice. The second build has no
        # class keywords but reuses the first class's namespace, and the
        # last class built is the one used to read documents. So the mark is
        # kept in the namespace, and the aliases are registered last.
        if kind:
            if format not in (2, 3):
                msg = (
                    f"{cls.__name__}: a kind declares the Zarr format its fields belong to, "
                    f"format=2 or format=3, got {format!r}"
                )
                raise TypeError(msg)
            cls._is_kind = True
            cls._format = format
        elif format is not None:
            msg = f"{cls.__name__}: only a kind declares a format; a definition of a kind has its kind's"
            raise TypeError(msg)
        for alias in cls.__dict__.get("field_aliases", ()):
            _FIELD_KINDS[alias] = cls

    def __post_init__(self) -> None:
        refusal = _malformed(self) or self._refusal()
        if refusal is not None:
            raise TypeError(refusal)
        try:
            _vet(self.configuration)
        except TypeError as error:
            msg = f"{self.name!r}: {error}"
            raise TypeError(msg) from error

    def __repr__(self) -> str:
        # Keep this short, because readings that contain definitions show
        # them. A full repr would show every function the definition holds,
        # with its address.
        return f"{type(self).__name__}(name={self.name!r})"

    def _refusal(self) -> str | None:
        """A description of what is wrong with the members a kind adds; None if nothing is wrong, or if the kind adds no members."""
        return None

    def _finite_numbers(self) -> bool:
        """Whether every number in a configuration this definition reads must be finite, as RFC 8259 requires. A kind whose configuration may hold user data overrides this."""
        return True

    @property
    def requires_configuration(self) -> bool:
        """Whether a document must include a configuration, meaning the TypedDict has a required key."""
        return len(typeddict_keys(self.configuration).required) != 0

    def _check_configuration(self, value: object, loc: Loc = ()) -> tuple[C | None, Problems]:
        """Type-check `value` as this definition's configuration, and check the envelope of each nested field.

        `zarr_metadata.typed_json.check` does only the type check. This
        method also checks the envelope of each metadata field that a
        member holds.
        """
        configuration, problems = _configuration_checked(
            value, self.configuration, loc, finite=self._finite_numbers()
        )
        return configuration, with_input(problems, value, loc)

    def read_configuration(self, value: object, loc: Loc = ()) -> tuple[C | None, Problems]:
        """Type-check `value`, then check it with the rules. Return the configuration if it is valid, and every problem.

        The rules run only on a configuration that passed the type check,
        including its bounds, and whose nested fields are well formed. Such
        a configuration contains only what its TypedDict admits. So a rule
        never receives a member of the wrong type, a member out of bounds,
        or a member the type does not allow. No scope reads the nested
        fields here, so the rules see none of them as read. A rule about a
        nested field (for example, a struct field whose type has values of
        varying size) finds nothing to check. `resolve` reads the field in a
        scope and runs every rule.
        """
        configuration, problems = self._check_configuration(value, loc)
        if configuration is None:
            return None, problems
        refused = ruled(self, lambda: self.rules(read_only(configuration), _nothing_nested()), loc)
        return (configuration if len(refused) == 0 else None), (
            *problems,
            *with_input(refused, value, loc),
        )


def _malformed(definition: Definition[Any]) -> str | None:
    """A description of what makes a hand-built definition unusable, checked before its configuration is examined; None if it is usable."""
    name = cast("object", definition.name)
    if not isinstance(name, str):
        return f"a definition's name is a string, got {name!r}"
    for member in _function_members(type(definition)):
        value = getattr(definition, member)
        if not callable(value):
            return f"{name!r}: {member} is a function, got {value!r}"
    kind = type(definition)
    filed, _ = kind.filed_under(name)
    bad = None if filed is None else kind.name_problem(name, ())
    if bad is not None:
        return (
            f"{name!r}: {bad.message}, so no document names it, and nothing would ever read "
            "with this definition"
        )
    return None


@functools.cache
def _function_members(kind: type[Definition[Any]]) -> tuple[str, ...]:
    """The members `kind` declares as functions: each member whose annotation is a `Callable`."""
    return tuple(
        member
        for member, annotation in get_type_hints(kind).items()
        if get_origin(annotation) is Callable
    )


def filed_under(
    kind: type[Definition[Any]], name: str
) -> tuple[str | None, dict[str, JSONValue] | None]:
    """Interpret a name used in a document: return the name its definition is filed under, and the configuration encoded in the name.

    A name is filed as itself and encodes nothing, except for raw bits:
    the data type `r16` is filed under `r*` and encodes `{"bits": 16}`.
    `r*` itself is filed under nothing, `(None, None)`, because no document
    uses it.
    """
    return kind.filed_under(name)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class WithFillValue(Definition[C]):
    """A definition of a type whose arrays have a fill value: the fill value's JSON shape, the rules for it, and its canonical spelling.

    This is not a kind. The data type kind of each format derives from it,
    so `fill_value_problems` can check a fill value for either format.

    `fill_value` is the JSON shape of a fill value, as an annotation. The
    checker reads it the same way it reads a configuration's members,
    including any range. `fill_value_rules` yields the spec violations in
    a fill value of that shape that the type cannot express. It receives
    the configuration, the nested fields as the scope read them, and the
    typed fill value. `fill_value_canonical` writes a fill value without
    problems in the single canonical spelling of its value. So two fill
    values are the same value of the type exactly when their canonical
    spellings are written the same way.
    """

    fill_value: object = JSONValue
    """The JSON shape of a fill value, as an annotation, such as `Int8FillValue`."""
    fill_value_rules: Callable[[C, Nested, Any], Iterable[ValidationProblem]] = no_rules
    """The spec violations in a fill value of that shape, located in the fill value."""
    fill_value_canonical: Callable[[C, Nested, Any], JSONValue] = fill_value_as_written
    """A fill value without problems, in the single canonical spelling of its value."""

    def _refusal(self) -> str | None:
        try:
            _fill_value_parser(self.fill_value)
        except TypeError as error:
            return f"{self.name!r}: fill_value: {error}"
        return None


@functools.cache
def _fill_value_parser(annotation: object) -> Parser:
    """The checker for a fill value's JSON shape, compiled once. Raises `TypeError` naming any part that no checker reads, or any metadata field in it."""
    return parser(annotation, _no_field)


def kind_of(definition: object) -> type[Definition[Any]] | None:
    """The kind of `definition`: the nearest class in its MRO declared with `kind=True`. None for a definition with no kind, which no scope can file, and for a value that is not a definition."""
    return _kind_of_class(type(definition))


@functools.lru_cache(maxsize=256)
def _kind_of_class(cls: type[object]) -> type[Definition[Any]] | None:
    """`kind_of` for a definition of class `cls`. It walks the MRO once per class. The cache is bounded, so a class created on each call is not kept forever."""
    return next((base for base in cls.__mro__ if _declares_kind(base)), None)


def format_of(kind: type[Definition[Any]]) -> Literal[2, 3]:
    """The Zarr format of the documents that contain fields of `kind`, as declared by the kind. Raises `TypeError` for a class that is not a kind."""
    found = as_kind(kind)._format  # pyright: ignore[reportPrivateUsage]
    if found is None:  # a kind cannot be declared without one
        msg = f"{kind!r} declares no format"
        raise TypeError(msg)
    return found


def _declares_kind(cls: type[object]) -> TypeGuard[type[Definition[Any]]]:
    """Whether `cls` itself was declared with `kind=True`. The mark is read from the class's own namespace, which a subclass does not share."""
    return vars(cls).get("_is_kind") is True and issubclass(cls, Definition)


@functools.lru_cache(maxsize=256)
def _kind_named(kind: object) -> type[Definition[Any]] | None:
    """The kind that `kind` refers to, with type arguments removed, or None. This is the cached lookup behind `as_kind`, run once per form of a kind (`CodecDefinition`, `CodecDefinition[Any]`), since a class is a kind by its declaration. The cache is bounded, so a class created on each call is not kept forever."""
    origin = get_origin(kind) or kind
    if isinstance(origin, type) and _declares_kind(origin):
        return origin
    return None


def as_kind(kind: object) -> type[Definition[Any]]:
    """The kind of metadata that `kind` refers to, with type arguments removed. Raises `TypeError` if it refers to no kind.

    A scope files definitions by kind, so a field is read as a kind
    (`CodecDefinition`, or `CodecDefinition[Any]`). It is never read as
    the base `Definition` or as a class that declares no kind, because
    nothing is filed under those, and a field read as one would never be
    checked.
    """
    try:
        found = _kind_named(kind)
    except TypeError:
        # The value is not hashable. No kind is unhashable.
        found = None
    if found is not None:
        return found
    msg = (
        f"{kind!r} is not a kind of metadata; read a field as one of the kinds of a format, "
        "such as CodecDefinition"
    )
    raise TypeError(msg)


def field_kind(annotation: object) -> type[Definition[Any]] | None:
    """The kind of metadata field held by a member annotated with `annotation` (`CodecDefinition` for `ZarrV3CodecField`), or None if the member holds no field."""
    if not is_alias(annotation):
        return None
    return _FIELD_KINDS.get(annotation)


class Context(Protocol):
    """A scope: the definitions in scope while metadata of one Zarr format is read.

    This is what reading a field needs from a scope: the Zarr format of the
    documents it reads, and which definition of a kind claims a name.
    `ZarrV3Context` and `ZarrV2Context` implement it, each with one slot
    per kind of field its format has, and so does any other object that
    provides these two. `resolve` reads a field in any `Context`.
    """

    @property
    def zarr_format(self) -> Literal[2, 3]: ...

    def claimant(self, kind: type[D], name: str) -> D | None: ...


@dataclass(frozen=True, slots=True)
class _NestedField:
    """A metadata field that the check found inside a configuration: its location, its kind, and its JSON.

    The checker returns this in place of the JSON wherever a field alias
    appears. `_checked` collects each one and puts its JSON back. It is
    carried in the value instead of being recorded separately, so a union
    branch that did not match leaves nothing behind.
    """

    loc: Loc
    kind: type[Definition[Any]]
    json: JSONValue
    alias: TypeAliasType
    """Whether the member that holds it accepts only codecs of static size."""


def _field(annotation: object) -> Parser | None:
    """The one checker shape that this module defines: a member that holds a metadata field.

    The value is checked only as a bare name or an object, and passed on.
    Whoever reads the field (`check` without a scope, `resolve` with one)
    checks the envelope and maps the name to a definition.
    """
    kind = field_kind(annotation)
    if kind is None or not is_alias(annotation):
        return None
    alias = annotation

    def parse(value: object, loc: Loc) -> Parsed:
        # The value is refined JSON, which the checker types only as
        # `object`. If it is not a field at all, the kind's envelope check
        # reports that.
        return _NestedField(loc, kind, cast("JSONValue", value), alias), ()

    return parse


def _vetting(annotation: object) -> Parser | None:
    """`_field`, but rejecting annotations that a definition's configuration cannot contain and still be read.

    A member typed with the envelope's own TypedDict
    (`ZarrV3MetadataFieldJSON`) is checked as plain JSON, so its name would
    never be mapped to a definition and its configuration would never be
    checked. A TypedDict that does not declare how undeclared keys are
    handled is open, so an undeclared key would never be reported.
    """
    if annotation is ZarrV3NamedConfigJSON:
        msg = (
            "a member typed ZarrV3MetadataFieldJSON checks as plain JSON, and is never read as "
            "a field; annotate it with the field alias of its kind: "
            + ", ".join(alias.__name__ for alias in _FIELD_KINDS)
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
            "extra_items= for what such a key holds, or closed=False to take any, "
            "on a typing_extensions.TypedDict"
        )
        raise TypeError(msg)
    return _field(annotation)


def _no_field(annotation: object) -> Parser | None:
    """A leaf checker that rejects a field alias. A fill value is a value of its data type and never contains a metadata field."""
    if is_alias(annotation) and field_kind(annotation) is not None:
        name = annotation.__name__
        msg = f"{name} holds a metadata field, and a fill value is a value of its data type"
        raise TypeError(msg)
    return None


@functools.cache
def _vet(configuration: type) -> None:
    """Reject a configuration that no definition could read with, and say what is wrong with it."""
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
    """A TypedDict's checker, compiled once, and whether a value of the TypedDict can contain a nested field."""

    parse: Parser
    nests: bool


@functools.cache
def _checker(shape: type) -> _Checker:
    """The checker for a TypedDict, compiled once. Raises `TypeError` naming any part that no checker reads."""
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
    """Check `value` as `shape`. Return the typed value (with each nested field restored as written), every problem, and the nested fields in order."""
    typed, found, nested = _typed(shape, value, loc)
    return _put_back(typed, _as_written, nested), found, nested


def _typed(
    shape: type, value: object, loc: Loc
) -> tuple[object, Problems, tuple[_NestedField, ...]]:
    """Check `value` as `shape`, leaving each nested field where the checker found it. Return the typed value, every problem, and the nested fields in order."""
    checker = _checker(shape)
    typed, found = checker.parse(value, loc)
    if not checker.nests:
        return typed, found, ()
    nested: list[_NestedField] = []
    _collect(typed, nested)
    return typed, found, tuple(nested)


def _collect(value: object, nested: list[_NestedField]) -> None:
    """Append to `nested` each nested field that the checker returned inside `value`, in order."""
    if isinstance(value, _NestedField):
        nested.append(value)
    elif is_tuple(value):
        for entry in value:
            _collect(entry, nested)
    elif is_object(value):
        for entry in value.values():
            _collect(entry, nested)


def _as_written(field: _NestedField) -> JSONValue:
    """A nested field restored as it was written."""
    return field.json


def _declared(field: _NestedField) -> JSONValue:
    """A nested field restored as it was written, except for envelope members that are not declared. Those are reported and left out, in the same way the checker leaves out a key that a closed TypedDict does not declare."""
    if not isinstance(field.json, Mapping):
        return field.json
    envelope = cast("Mapping[str, JSONValue]", field.json)
    return {key: value for key, value in envelope.items() if key in ENVELOPE_KEYS}


def _put_back(
    value: object, put: Callable[[_NestedField], JSONValue], nested: Sequence[_NestedField]
) -> object:
    """Replace each of `nested` (the fields the checker returned inside `value`) with the result of `put`. Return `value` unchanged when there are none."""
    if len(nested) == 0:
        return value
    return _putting_back(value, put)


def _putting_back(value: object, put: Callable[[_NestedField], JSONValue]) -> object:
    """`_put_back` for a value that contains nested fields: a recursive walk."""
    if isinstance(value, _NestedField):
        return put(value)
    if is_tuple(value):
        return tuple(_putting_back(entry, put) for entry in value)
    if is_object(value):
        return {key: _putting_back(entry, put) for key, entry in value.items()}
    return value


def _usable(problems: Sequence[ValidationProblem]) -> bool:
    """Whether a value with these problems can still be read. An unknown key is allowed. Any other problem is not."""
    return all(found.kind == "unknown_key" for found in problems)


def read_only(configuration: C) -> C:
    """`configuration` in the form a definition's functions receive it: a read-only view of the field's own configuration, so a function that assigns a member fails.

    This function makes only the top-level members read-only. The value a
    member holds (for example, an object in a struct's `fields`) is passed
    as is. While a configuration is being read, that value can be mutable,
    and a function that writes into it changes the configuration. An
    accepted field's configuration is already read-only at every level.
    """
    return cast("C", MappingProxyType(cast("Mapping[str, JSONValue]", configuration)))


def asked(definition: Definition[Any], what: str, ask: Callable[[], T], at: Loc | None = None) -> T:
    """Return the result of `ask`, which calls the function `what` of `definition`.

    A definition's functions are the extension author's code. If one
    raises an error, a note is added that says which definition's
    function raised it and, when known, where the reader was reading.
    """
    try:
        return ask()
    except Exception as error:
        error.add_note(_raised_by(definition, what, at))
        raise


def _raised_by(definition: Definition[Any], what: str, at: Loc | None) -> str:
    """The note added to an error raised by the function `what` of `definition`: which definition's function raised it and, when known, where the reader was reading."""
    where = "" if at is None else f", reading {at!r}"
    return f"raised by the {what} of {definition.name!r}{where}"


def ruled(
    definition: Definition[Any], ask: Callable[[], Iterable[ValidationProblem]], at: Loc
) -> Problems:
    """The problems found by `ask`, which calls `definition`'s rules, located under `at`.

    Rules are the extension author's code. Their output is checked to have
    the declared type. If they raise an error, a note is added that says
    which definition's rules raised it and where they were reading.
    """
    try:
        found = tuple(cast("Iterable[object]", ask()))
    except Exception as error:
        error.add_note(_raised_by(definition, "rules", at))
        raise
    if len(found) == 0:
        return ()
    for item in found:
        if not isinstance(item, ValidationProblem):
            msg = f"{definition.name!r}: its rules yield ValidationProblem values, got {item!r}"
            raise TypeError(msg)
    return _located(at, cast("tuple[ValidationProblem, ...]", found))


def _located(prefix: Loc, problems: Iterable[ValidationProblem]) -> Problems:
    """`problems`, each located under `prefix`. The input may be lazy, because a kind's `envelope_problems` may return an iterator."""
    found = tuple(problems)
    if len(found) == 0:
        return ()
    return tuple(dataclasses.replace(each, loc=(*prefix, *each.loc)) for each in found)


def _envelope(field: _NestedField) -> Problems:
    """The problems with a nested field's envelope, located at the field."""
    return _located(field.loc, field.kind.envelope_problems(field.json))


def _configuration_checked(
    value: object, shape: type[T], loc: Loc = (), *, finite: bool = True
) -> tuple[T | None, Problems]:
    """Check `value` as `shape`, as `typed_json.check` does, and check the envelope of each nested field.

    A definition's `read_configuration` starts with this step. A member
    typed with a field alias holds a metadata field, and its envelope is
    checked the same way as in a document. A `must_understand` of `false`
    is a problem of the configuration, and no value is returned. An extra
    member is an unknown key: it is reported and left out, and the value is
    still returned. This matches how the checker reports and leaves out a
    key that a closed TypedDict does not declare. A number that is not
    finite is a problem unless `finite` is false, for a configuration that
    holds user data; the TypedDict still decides where such a number may
    be.
    """
    refined, problems = (refine_json if finite else refine_user_data)(value, loc)
    if len(problems) != 0:
        return None, problems
    typed, found, nested = _typed(shape, refined, loc)
    problems = (*found, *(problem for field in nested for problem in _envelope(field)))
    if not _usable(problems):
        return None, problems
    return cast("T", _put_back(typed, _declared, nested)), problems


def named_configuration(
    value: object,
) -> tuple[str | None, Mapping[str, object] | None, Problems]:
    """Split a metadata field into `(name, configuration, problems)`.

    A field is a bare name, or an object that contains a name. A `None`
    name means the value is not a metadata field at all. A `None`
    configuration means the bare spelling was used, or the key was left
    out. The only problem reported is a configuration that is present but
    not an object, at `("configuration",)`.
    """
    if isinstance(value, str):
        return value, None, ()
    if not is_object(value):
        return None, None, ()
    entry = value
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, ()
    if "configuration" not in entry:
        return name, None, ()
    configuration = entry["configuration"]
    if not isinstance(configuration, Mapping):
        return (
            name,
            None,
            problem(("configuration",), f"expected an object, got {shown(configuration)}"),
        )
    return name, cast("Mapping[str, object]", configuration), ()


def _nothing_nested() -> Nested:
    """The nested fields of a field that contains no read fields: an empty mapping."""
    return {}


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptedField(Generic[D]):
    """A field that a definition in scope read: the name it is written with, the definition, and the configuration the definition allowed.

    A problem with the envelope (an extra member, a `must_understand` of
    `false`) is reported with the field, and the field is still read. The
    same applies to a problem in a nested field, which belongs to that
    nested field, as `nested` shows. Two fields are equal when they read
    the same, however each was spelled, as compared by `field_key`. For
    example, `"bytes"` and `{"name": "bytes"}` are one field. So are a
    blosc with and without the `typesize` that `noshuffle` ignores, because
    the definition's `canonical` removes it. Equal fields have equal
    hashes.
    """

    json: JSONValue
    """The field as written, refined (arrays as tuples). It is not used for equality."""
    name: str
    """The name the field is written with, for example `"r16"`, even though its definition is filed under `r*`."""
    definition: D
    """The definition that read the field."""
    configuration: Mapping[str, JSONValue]
    """The configuration, type-checked and allowed by the rules. For raw bits, this is the configuration encoded in the name.

    Each nested field is written as that field's `to_json` writes it, so
    the configuration shows what was read, however it was spelled. For
    example, a shard's `"crc32c"` and `{"name": "crc32c"}` are the same
    index codec.
    """
    nested: Nested = dataclasses.field(default_factory=_nothing_nested)
    """The fields in the configuration, each as the scope read it, keyed by location in the configuration.

    For example, a struct's field types are at `("fields", 0, "data_type")`
    and a shard's codecs are at `("codecs", 0)`. A definition's functions
    use this to inspect the fields inside its own configuration.
    """
    read_as: type[Definition[Any]] = dataclasses.field(init=False, repr=False)
    """The kind of metadata the field was read as: the kind of its definition."""

    def __eq__(self, other: object) -> bool:
        if not is_field(other):
            return NotImplemented
        return field_key(self) == field_key(other)

    def __hash__(self) -> int:
        return hash(field_key(self))

    def __post_init__(self) -> None:
        # Check the annotations at runtime. A field built by hand (an
        # extension may do this) fails here, not later in a function that
        # trusts it.
        definition = cast("object", self.definition)
        kind = kind_of(definition)
        refusal = (
            f"a field read is read by a definition of a kind, got {definition!r}"
            if kind is None
            else _misread(definition, kind, self.name)
        )
        if refusal is not None:
            raise TypeError(refusal)
        object.__setattr__(self, "read_as", kind)
        # Everything the field returns is read-only at every level, so a
        # field cannot reach a state where its key, `==` and `refines`
        # disagree.
        object.__setattr__(self, "json", frozen(self.json))
        object.__setattr__(self, "configuration", frozen(self.configuration))
        object.__setattr__(self, "nested", MappingProxyType(dict(self.nested)))

    def __reduce__(self) -> tuple[Callable[..., AcceptedField[Any]], tuple[object, ...]]:
        # Read-only views cannot be pickled, so pickle the field from its constructor arguments.
        return _accepted_field, (
            copied(self.json),
            self.name,
            self.definition,
            copied(self.configuration),
            dict(self.nested),
        )

    def to_json(self) -> JSONValue:
        """The field as a document writes it, in a form every reader accepts. The configuration is as read, and the result shares no objects with the field.

        The envelope uses the shortest form every reader accepts. A data
        type with nothing to configure is written as its bare name, as core
        data types have been written since Zarr v3.0. Any other field is an
        object, `{"name": ...}`, because a Zarr v3.0 reader does not accept a
        bare name in `codecs`
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L585-L592).
        A name that encodes its configuration, as the raw bits name does, is
        written alone.
        """
        return copied(written_json(self))


@dataclass(frozen=True, slots=True, kw_only=True)
class UnclaimedField:
    """A field that nothing in scope claims: an extension that the scope does not check. Accepting such fields is what keeps the format open.

    It is equal to another unclaimed field when both are written with the
    same name and configuration, however each was spelled. Nothing in
    scope interprets the configuration, so it is compared as JSON text, as
    `field_key` defines.
    """

    json: JSONValue
    """The field as written, refined (arrays as tuples). It is not used for equality."""
    name: str
    """The name that nothing in scope claims."""
    read_as: type[Definition[Any]]
    """The kind of metadata the field was read as: the kind a claiming definition would have."""
    configuration: Mapping[str, JSONValue] = dataclasses.field(init=False)
    """The configuration as written, not checked; empty when no configuration is written."""

    def __eq__(self, other: object) -> bool:
        if not is_field(other):
            return NotImplemented
        return field_key(self) == field_key(other)

    def __hash__(self) -> int:
        return hash(field_key(self))

    def __post_init__(self) -> None:
        # Check the annotations at runtime. Remove type arguments from
        # `read_as`, as `resolve` does.
        object.__setattr__(self, "read_as", as_kind(self.read_as))
        name = cast("object", self.name)
        if not isinstance(name, str) or not self.read_as.well_named(name):
            msg = (
                "a field nothing in scope claims is named as a document names a "
                f"{self.read_as.label}, got {name!r}"
            )
            raise TypeError(msg)
        _, written, _ = self.read_as.named_configuration(self.json)
        configuration: Mapping[str, object] = {} if written is None else written
        object.__setattr__(self, "json", frozen(self.json))
        object.__setattr__(
            self, "configuration", frozen(cast("Mapping[str, JSONValue]", configuration))
        )

    def __reduce__(self) -> tuple[Callable[..., UnclaimedField], tuple[object, ...]]:
        # Read-only views cannot be pickled, so pickle the field from its constructor arguments.
        return _unclaimed_field, (copied(self.json), self.name, self.read_as)

    @property
    def definition(self) -> None:
        """The definition that read the field. Always None, because no definition read it."""
        return None

    @property
    def nested(self) -> Nested:
        """The nested fields as the scope read them. Always empty, because nothing read the configuration."""
        return _nothing_nested()

    def to_json(self) -> JSONValue:
        """The field as a document writes it, sharing no objects with the field. The configuration is as written, in the envelope form every reader accepts, as `AcceptedField.to_json` writes it."""
        return copied(written_json(self))


@dataclass(frozen=True, slots=True, kw_only=True)
class RefusedField(Generic[D]):
    """A field that could not be read: it is not a field at all, not JSON, or rejected by the definition that claims its name. Its problems explain why."""

    json: JSONValue | UNSET
    """The field as written, refined (arrays as tuples). `UNSET` when it is not JSON, which no document can contain."""
    name: str | None
    """The name the field is written with; None when it has no name."""
    read_as: type[Definition[Any]]
    """The kind of metadata the field was read as."""
    definition: D | None = None
    """The definition that claims the name and rejected the field; None when nothing in scope claims it, or when the field has no name."""
    nested: Nested = dataclasses.field(default_factory=_nothing_nested)
    """The nested fields, each as the scope read it; empty when the configuration was not checked against its TypedDict."""

    def __eq__(self, other: object) -> bool:
        if not is_field(other):
            return NotImplemented
        return field_key(self) == field_key(other)

    def __hash__(self) -> int:
        return hash(field_key(self))

    def __post_init__(self) -> None:
        # Check the annotations at runtime. Remove type arguments from
        # `read_as`, as `resolve` does.
        kind = as_kind(self.read_as)
        object.__setattr__(self, "read_as", kind)
        definition = cast("object", self.definition)
        refusal = None if definition is None else _misread(definition, kind, self.name)
        if refusal is not None:
            raise TypeError(refusal)
        if self.json is not UNSET:
            object.__setattr__(self, "json", frozen(self.json))
        object.__setattr__(self, "nested", MappingProxyType(dict(self.nested)))

    def __reduce__(self) -> tuple[Callable[..., RefusedField[Any]], tuple[object, ...]]:
        # Read-only views cannot be pickled, so pickle the field from its constructor arguments.
        return _refused_field, (
            UNSET if self.json is UNSET else copied(self.json),
            self.name,
            self.read_as,
            self.definition,
            dict(self.nested),
        )


ResolvedField = TypeAliasType(
    "ResolvedField", "AcceptedField[D] | UnclaimedField | RefusedField[D]", type_params=(D,)
)
"""One metadata field as a scope read it: accepted by the definition that claims its name, unclaimed, or refused."""


def _accepted_field(
    json: JSONValue,
    name: str,
    definition: Definition[Any],
    configuration: Mapping[str, JSONValue],
    nested: Nested,
) -> AcceptedField[Any]:
    """Rebuild an accepted field from its pickled arguments."""
    return AcceptedField(
        json=json, name=name, definition=definition, configuration=configuration, nested=nested
    )


def _unclaimed_field(json: JSONValue, name: str, read_as: type[Definition[Any]]) -> UnclaimedField:
    """Rebuild an unclaimed field from its pickled arguments."""
    return UnclaimedField(json=json, name=name, read_as=read_as)


def _refused_field(
    json: JSONValue | UNSET,
    name: str | None,
    read_as: type[Definition[Any]],
    definition: Definition[Any] | None,
    nested: Nested,
) -> RefusedField[Any]:
    """Rebuild a refused field from its pickled arguments."""
    return RefusedField(json=json, name=name, read_as=read_as, definition=definition, nested=nested)


_FIELDS: Final = (AcceptedField, UnclaimedField, RefusedField)
"""The three possible results of reading a field in a scope."""


def _misread(definition: object, kind: type[Definition[Any]], name: object) -> str | None:
    """A description of why `definition` cannot be the definition that read a field named `name` as a `kind`; None when it can."""
    if not isinstance(definition, kind):
        return f"a field read as a {kind.__name__} is read by one, got {definition!r}"
    filed = definition.name
    if not isinstance(name, str) or filed_under(kind, name)[0] != filed:
        return f"a field named {name!r} is read by the definition filed under it, got {filed!r}"
    return None


def field_key(field: ResolvedField[Any]) -> tuple[object, ...]:
    """The key that `==` and `hash` compare for a field. It represents what the field means, not how it was spelled.

    An accepted field is compared by the definition that read it, by its
    configuration in the definition's canonical spelling (as
    `canonical_of` writes it) as JSON text, and by the key of each nested
    field. An unclaimed field is compared by its name and its
    configuration as written, as JSON text, because nothing interprets it.
    A refused field is compared by what was written and by the definition
    that rejected it. So two fields are equal when the parts a reader
    understands read the same, and the parts no reader interprets are
    written the same way.
    """
    if isinstance(field, AcceptedField):
        definition = cast("Definition[Any]", field.definition)
        configuration: JSONValue = dict(field.configuration)
        # Each nested field is compared by its own key, so its location is
        # cleared before the definition's `canonical` sees the rest, in the
        # same order as `_canonical_field`. `canonical` simplifies only the
        # definition's own members.
        for loc in field.nested:
            configuration = _replaced(configuration, loc, None)
        view = read_only(cast("Mapping[str, JSONValue]", configuration))
        spelled = asked(
            definition,
            "canonical",
            lambda: json_text(cast("JSONValue", dict(definition.canonical(view)))),
        )
        return ("read", definition, spelled, _nested_key(field.nested))
    if isinstance(field, UnclaimedField):
        return ("unclaimed", field.read_as, field.name, json_text(written_json(field)))
    return (
        "refused",
        field.read_as,
        field.name,
        field.definition,
        UNSET if field.json is UNSET else json_text(field.json),
        _nested_key(field.nested),
    )


def own_key(field: AcceptedField[Any]) -> tuple[object, ...]:
    """The part of `field_key` for an accepted field that excludes nested fields: the definition and the canonical spelling of its own members."""
    return field_key(field)[:3]


def _nested_key(nested: Nested) -> tuple[tuple[Loc, tuple[object, ...]], ...]:
    """The key of each nested field, paired with its location."""
    return tuple((loc, field_key(inner)) for loc, inner in nested.items())


def document_json(field: ResolvedField[Any]) -> JSONValue | UNSET:
    """A field as a document writes it, using the field's own values: as `to_json` writes it, or as originally written for a refused field. `UNSET` for a refused field that was not JSON.

    A writer serializes this without changing it, so it is not copied.
    `to_json` is a copy of this.
    """
    if isinstance(field, RefusedField):
        return field.json
    return written_json(field)


def written_json(field: AcceptedField[Any] | UnclaimedField) -> JSONValue:
    """A field that a scope accepted or left unclaimed, as a document writes it. This is `document_json` for a field that is JSON."""
    kind = field.read_as
    if isinstance(field, UnclaimedField):
        return kind.unclaimed_json(field.json, field.name, field.configuration)
    if kind.filed_under(field.name)[1] is not None:
        return kind.envelope_json(field.name, {})
    return kind.envelope_json(field.name, field.configuration)


Nested: TypeAlias = Mapping[Loc, ResolvedField[Any]]
"""The fields in a configuration, each as the scope read it, keyed by location in the configuration."""


def is_field(value: object) -> TypeGuard[ResolvedField[Any]]:
    """Whether `value` is a field that a scope read: `AcceptedField`, `UnclaimedField` or `RefusedField`."""
    return isinstance(value, _FIELDS)


def held(field: ResolvedField[D] | UNSET) -> AcceptedField[D] | UnclaimedField:
    """Return `field`, which must come from a read that found no problems, as `AcceptedField` or `UnclaimedField`. Raises `TypeError` for a refused field or one never read, which such a read cannot produce."""
    if isinstance(field, (AcceptedField, UnclaimedField)):
        return field
    msg = f"expected a field a read found nothing wrong with, got {field!r}"
    raise TypeError(msg)


def fields_of(
    resolved: ResolvedField[Any], loc: Loc = ()
) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
    """Yield `resolved` (a field a scope read) with its location, then each nested field recursively, each with its location.

    `loc` is the location of `resolved`. A nested field is located inside
    the configuration, at the kind's `configuration_loc` followed by the
    field's place, the same way `resolve` locates problems. Each field's
    nested fields are its `nested`. A field whose configuration was not
    checked has none.
    """
    yield loc, resolved
    for place, inner in resolved.nested.items():
        yield from fields_of(inner, (*resolved.read_as.configuration_loc(loc), *place))


def with_problems(
    fields: Iterable[tuple[Loc, ResolvedField[Any]]], problems: Sequence[ValidationProblem]
) -> Iterator[tuple[Loc, ResolvedField[Any], Problems]]:
    """Yield each of `fields` with its location and the problems in `problems` located inside it, including inside its nested fields.

    `fields` and `problems` must come from the same read: a reading's
    `fields()` and `problems`, or `fields_of` a field and the problems that
    `resolve` returned with it. A field's problems include those found when
    it was read, and those the document found with it in its position (its
    place in the pipeline, the chunk it receives, the array's shape). So a
    field with no problems is valid in that position, and `canonical_of`
    can write it. This groups the problems at every depth, as zod's
    `treeifyError` groups issues: a problem with a shard's inner codec
    belongs to the inner codec and to the shard. Each field comes before
    its nested fields, in the order of `fields_of`, so the last field that
    lists a problem is the innermost field containing it. A problem
    outside every field (with the fill value or the shape) is not listed
    under any field.
    """
    located = list(fields)
    held: dict[Loc, list[ValidationProblem]] = {loc: [] for loc, _ in located}
    for found in problems:
        for depth in range(len(found.loc) + 1):
            holder = held.get(found.loc[:depth])
            if holder is not None:
                holder.append(found)
    for loc, field in located:
        yield loc, field, tuple(held[loc])


def configuration_of(resolved: ResolvedField[Any], definition: Definition[C]) -> C | None:
    """The configuration of `resolved`, typed as `definition` declares it, if `definition` read it.

    `AcceptedField` stores its configuration as a plain mapping. When
    called with the definition that read the field, or one equal to it
    (as with a pickled field), this returns the same mapping typed as the
    TypedDict. Returns None when another definition read the field, or no
    definition did.
    """
    if not isinstance(resolved, AcceptedField) or resolved.definition != definition:
        return None
    return cast("C", resolved.configuration)


def fill_value_problems(data_type: ResolvedField[F], value: object, loc: Loc = ()) -> Problems:
    """The problems with `value` as a fill value of `data_type` (a data type field a scope read).

    `value` is refined to JSON first. If it is not JSON, that is reported
    first, whatever the data type. It is then checked against the JSON
    shape that the data type's definition declares, and by the fill value
    rules, in the same way `read_configuration` reads a configuration: a
    key the shape does not declare is reported and left out, and the rules
    still check the rest. The rules see the nested fields of the
    configuration as the scope read them, so a struct checks each field's
    fill value against that field's own type. If the scope did not read
    the data type (it is out of scope or invalid), a JSON fill value is not
    checked. `loc` prefixes every problem.
    """
    refined, problems = refine_json(value, loc)
    if len(problems) != 0 or not isinstance(data_type, AcceptedField):
        return with_input(problems, value, loc)
    definition, configuration = data_type.definition, data_type.configuration
    typed, problems = _fill_value_parser(definition.fill_value)(refined, loc)
    if not _usable(problems):
        return with_input(problems, value, loc)
    refused = ruled(
        definition,
        lambda: definition.fill_value_rules(read_only(configuration), data_type.nested, typed),
        loc,
    )
    return with_input((*problems, *refused), value, loc)


def canonical_fill_value(data_type: ResolvedField[F], value: object) -> JSONValue | UNSET:
    """`value` (a fill value of `data_type`, a data type field a scope read) in the single canonical spelling of its value; `UNSET` when it has a problem.

    The spelling comes from the data type's `fill_value_canonical`. Two
    fill values of a data type are the same value exactly when their
    canonical spellings are written the same way: the same JSON as
    `json.dumps` writes it. `==` is not the same test, because it treats
    `-0.0` as equal to `0.0`. A fill value with a problem found by
    `fill_value_problems` has no canonical spelling, just as `canonical_of`
    returns none for a field with a problem. The result is `UNSET` and not
    `None`, because `None` is the JSON `null`, which is a valid fill value
    for a data type the scope did not read (such a data type keeps a fill
    value as written).
    """
    if len(fill_value_problems(data_type, value)) != 0:
        return UNSET
    refined, _ = refine_json(value, ())
    return spelled_canonically(data_type, refined)


def spelled_canonically(data_type: ResolvedField[F], value: JSONValue) -> JSONValue:
    """`value` (a fill value of `data_type` with no problems) in its canonical spelling, as `canonical_fill_value` returns it, without checking it again.

    The data type's `fill_value_canonical` is the extension author's code.
    Its result is checked to be JSON. If it raises an error, a note is
    added that says which data type's canonical spelling raised it.
    """
    if not isinstance(data_type, AcceptedField):
        return value
    definition, configuration = data_type.definition, data_type.configuration
    spelled = asked(
        definition,
        "fill_value_canonical",
        lambda: cast(
            "object",
            definition.fill_value_canonical(read_only(configuration), data_type.nested, value),
        ),
    )
    refined, problems = refine_json(spelled, ())
    if len(problems) != 0:
        msg = (
            f"{definition.name!r}: its fill_value_canonical gives JSON, got {spelled!r}: "
            f"{problems[0].message}"
        )
        raise TypeError(msg)
    return refined


def resolve(
    data: object, kind: type[D], context: Context, loc: Loc = ()
) -> tuple[ResolvedField[D], Problems]:
    """Read `data` (one metadata field) as a `kind` in `context`. Return the result of the reading, and every problem.

    This runs all three steps for one field. `data` is refined to JSON and
    its envelope is checked. An extra member, a `configuration` that is not
    an object, and a `must_understand` that is not a boolean or is `false`
    are each a problem. The name is mapped to a definition in `context`.
    The configuration is checked against its TypedDict and by its rules.
    Each nested field found by the check is read the same way, in the same
    scope. A problem in a nested field belongs to that field and is
    reported at its location, as with a document's fields. The result is
    an `AcceptedField` read by the definition that claims the name; an
    `UnclaimedField` when nothing in scope claims the name (an extension
    with no model, not checked, which is what keeps the format open); or a
    `RefusedField`, with the problems that explain why. `loc` prefixes
    every problem. `kind` must be one of the kinds of a format (for v3,
    those `ZarrV3Kind` names; for v2, those `ZarrV2Kind` names), with or
    without type arguments, and `context` a scope of that format. Anything
    else raises `TypeError`.
    """
    asked = as_kind(kind)
    name = asked.named_configuration(data)[0]
    definition = (
        None if name is None or not asked.well_named(name) else context.claimant(asked, name)
    )
    refined, problems = asked.refined(data, loc, definition)
    return resolve_refined(data, refined, problems, kind, context, loc)


def resolve_refined(
    data: object,
    refined: JSONValue | None,
    problems: Problems,
    kind: type[D],
    context: Context,
    loc: Loc = (),
) -> tuple[ResolvedField[D], Problems]:
    """`resolve` for `data` that the caller has already refined. `refined` and `problems` are the result of `refine_json(data, loc)`.

    This is for a reader that refines a document's members once and keeps
    the result (the document a model holds is built from it), instead of
    refining each field again here. `problems` shows whether `data` is
    JSON. If it contains any problems, `refined` is not used.
    """
    asked = as_kind(kind)
    scope_format = context.zarr_format
    if scope_format != format_of(asked):
        msg = (
            f"a {asked.label} is a field of a Zarr v{format_of(asked)} document, read in a scope "
            f"of that format, got a v{scope_format} scope"
        )
        raise TypeError(msg)
    if len(problems) != 0:
        # The value is not JSON, so it is not read. If it has a name, the
        # name still determines which definition claims it. A name that is
        # not a valid extension name under the spec is a problem here, as on
        # the other path, and no definition is looked up for it.
        name = asked.named_configuration(data)[0]
        bad = None if name is None else asked.name_problem(name, asked.name_loc(loc))
        claimant = None if name is None or bad is not None else context.claimant(asked, name)
        refused = RefusedField(json=UNSET, name=name, read_as=asked, definition=claimant)
        found = problems if bad is None else (bad, *problems)
        return cast("ResolvedField[D]", refused), with_input(found, data, loc)
    resolved, found = _resolve_field(refined, asked, context, loc)
    return cast("ResolvedField[D]", resolved), with_input(found, data, loc)


def _resolve_field(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[ResolvedField[Definition[Any]], Problems]:
    """Check the envelope of a refined field, then read the field.

    The result of the reading depends only on the configuration. An extra
    member or a `must_understand` of `false` does not affect the
    configuration, so it is reported alongside the field that was read,
    and later layers can still check that field.
    """
    envelope = _located(loc, kind.envelope_problems(data))
    resolved, found = _read(data, kind, context, loc)
    return resolved, (*envelope, *found)


def _read(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[ResolvedField[Definition[Any]], Problems]:
    name, given, malformed = kind.named_configuration(data)
    if name is None:
        return RefusedField(json=data, name=None, read_as=kind), ()
    if not kind.well_named(name):
        # The envelope check that every reader runs first reports this. No
        # definition is looked up for the name.
        return RefusedField(json=data, name=name, read_as=kind), ()
    definition = context.claimant(kind, name)
    if len(malformed) != 0:
        # The configuration is not an object, which the envelope problems
        # report. The name still determines which definition claims the
        # field.
        return RefusedField(json=data, name=name, read_as=kind, definition=definition), ()
    if definition is None:
        return UnclaimedField(json=data, name=name, read_as=kind), ()
    _, carried = kind.filed_under(name)
    if carried is not None:
        return _read_carried(data, name, kind, definition, given, carried, loc)
    at = kind.configuration_loc(loc)
    if given is None and definition.requires_configuration:
        missing = problem(at, f"{name!r} requires a configuration", "missing_key")
        return RefusedField(json=data, name=name, read_as=kind, definition=definition), missing
    typed, found, nested = _typed(definition.configuration, {} if given is None else given, at)
    # Read the nested fields first, each one stack frame deeper, so the
    # rules see them as the scope read them. Each is put back as a document
    # writes it, so the configuration shows what was read, however each was
    # spelled. Their problems are reported after the problems from the
    # rules.
    within: dict[Loc, ResolvedField[Any]] = {}
    written: dict[Loc, JSONValue] = {}
    inside: list[ValidationProblem] = []
    for field in nested:
        inside.extend(_envelope(field))
        inner, found_inside = _read(field.json, field.kind, context, field.loc)
        within[field.loc[len(at) :]] = inner
        # The input was refined JSON, so the result is JSON, or a field
        # refused for its contents, never for not being JSON.
        written[field.loc] = cast("JSONValue", document_json(inner))
        inside.extend(found_inside)
        inside.extend(_sized(field, inner))
    # The rules may look up a nested field by its name, so they run only
    # when every nested field has a name. Any other problem with a nested
    # field belongs to that field and is reported at its location, as with
    # a document's fields.
    sound = _usable(found) and all(_named(field) for field in nested)
    configuration = (
        cast(
            "Mapping[str, JSONValue]",
            _put_back(typed, lambda field: written[field.loc], nested),
        )
        if sound
        else None
    )
    own = list(found)
    if configuration is not None:
        own.extend(
            ruled(definition, lambda: definition.rules(read_only(configuration), within), at)
        )
    if configuration is None or not _usable(own):
        refused = RefusedField(
            json=data, name=name, read_as=kind, definition=definition, nested=within
        )
        return refused, (*own, *inside)
    read = AcceptedField(
        json=data, name=name, definition=definition, configuration=configuration, nested=within
    )
    return read, (*own, *inside)


def _named(field: _NestedField) -> bool:
    """Whether a nested field has a name, and its configuration (if any) is an object."""
    name, _, malformed = field.kind.named_configuration(field.json)
    return name is not None and len(malformed) == 0


def _read_carried(
    data: JSONValue,
    name: str,
    kind: type[Definition[Any]],
    definition: Definition[Any],
    given: Mapping[str, object] | None,
    carried: Mapping[str, JSONValue],
    loc: Loc,
) -> tuple[ResolvedField[Definition[Any]], Problems]:
    """Read a field whose name encodes its configuration (raw bits, such as `r16`) with the definition its name is filed under.

    The document wrote a name, so a problem with the configuration encoded
    in the name (such as a size that is not a positive multiple of 8) is a
    problem of the field. A configuration written next to the name must be
    empty, so each member in it is an undeclared key.
    """
    _, beside, _ = _checked(
        EmptyConfiguration,
        {} if given is None else given,
        kind.configuration_loc(loc),
    )
    configuration, judged = definition.read_configuration(carried)
    # A problem with the configuration encoded in the name belongs to the
    # field. It is located at the field, where the name is, and the
    # `input` and `ctx` that described a configuration member do not apply
    # to the name, so they are cleared.
    problems = (
        *beside,
        *(dataclasses.replace(found, loc=loc, input=UNSET, ctx={}) for found in judged),
    )
    # A name is a single token. An undeclared part of the encoded
    # configuration cannot be left out, as an extra key next to the name
    # can, so any problem with the encoded configuration refuses the field.
    if configuration is None or not _usable(beside) or len(judged) != 0:
        refused = RefusedField(json=data, name=name, read_as=kind, definition=definition)
        return refused, problems
    return AcceptedField(
        json=data, name=name, definition=definition, configuration=configuration
    ), problems


def _sized(field: _NestedField, inner: ResolvedField[Any]) -> Problems:
    """The problems with the definition that read `inner` in the member that holds `field`, as the kind of `field` reports them for its alias.

    A name that nothing in scope claims is not checked. Nothing is known
    about it.
    """
    definition = inner.definition
    if definition is None:
        return ()
    return field.kind.alias_problems(field.alias, definition, field.loc)


def canonicalize(
    data: object, kind: type[D], context: Context, loc: Loc = ()
) -> tuple[JSONValue | None, Problems]:
    """`data` (one metadata field) in its simplest equivalent spelling, and every problem.

    Only a field without problems has a simplest spelling. A simplest
    spelling expresses what the author wrote in fewer words. A key the
    TypedDict does not declare, an extra envelope member, or a
    `must_understand` of `false` is something the author wrote that the
    simplest spelling would erase. So a field with any problem returns
    None, with its problems. Otherwise each nested field is written in its
    own simplest spelling, and then the definition's `canonical` handles
    the rest. Its result is checked again, so a `canonical` that returns
    an invalid configuration raises `ValueError`, which indicates a bug in
    the definition, not in the field. The envelope uses the shortest form
    every reader accepts. A data type with nothing to configure is written
    as its bare name, as core data types have been written since Zarr
    v3.0. Any other field is an object, `{"name": ...}`, because a Zarr
    v3.0 reader does not accept a short-hand name in `codecs`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L585-L592).
    There is no `must_understand`, because leaving it out means `true`.
    A name that nothing in scope claims keeps the configuration it was
    written with, because only its own definition could decide how to
    simplify it. For a field that a scope has already read, use
    `canonical_of` with its problems.
    """
    resolved, problems = resolve(data, kind, context, loc)
    return canonical_of(resolved, problems), problems


def canonical_of(
    resolved: ResolvedField[Any], problems: Sequence[ValidationProblem]
) -> JSONValue | None:
    """`resolved` (a field a scope read) in its simplest equivalent spelling, as `canonicalize` writes it; None when it has a problem.

    `problems` are the field's problems, as returned by `resolve`, or as
    `with_problems` returns them for each field of a reading. Only a field
    with no problems has a simplest spelling. A simpler spelling of a
    field with a problem would erase what its author wrote (such as a key
    its TypedDict does not declare), or would write something invalid.
    The result is built from what the scope read, without reading the
    field again, and is plain JSON that shares nothing with the field.
    """
    if len(problems) != 0:
        return None
    simplest = _simplest(resolved)
    return None if simplest is None else copied(simplest)


def _simplest(field: ResolvedField[Any]) -> JSONValue | None:
    """A field in its simplest spelling; None when the field or one of its nested fields was refused, because a refused field has no simplest spelling."""
    if isinstance(field, AcceptedField):
        return _canonical_field(field)
    if isinstance(field, UnclaimedField):
        return field.to_json()
    return None


def _canonical_field(resolved: AcceptedField[Any]) -> JSONValue | None:
    """An accepted field in its simplest equivalent spelling: nested fields first, then its own members. None when a nested field was refused."""
    definition, name = resolved.definition, resolved.name
    configuration: JSONValue = dict(resolved.configuration)
    for loc, inner in resolved.nested.items():
        simplest = _simplest(inner)
        if simplest is None:
            return None
        configuration = _replaced(configuration, loc, simplest)
    # Every function of a definition receives this read-only view. The
    # result of `canonical` (the view itself when nothing is simplified) is
    # converted to a dict.
    view = read_only(cast("Mapping[str, JSONValue]", configuration))
    simplified = asked(
        definition,
        "canonical",
        lambda: dict(cast("Mapping[str, JSONValue]", definition.canonical(view))),
    )
    _, refused = definition.read_configuration(simplified)
    if len(refused) != 0:
        msg = (
            f"{definition.name!r}: its canonical gave {simplified!r}, which does not hold: "
            f"{list(refused)!r}"
        )
        raise ValueError(msg)
    carrying = definition.carrying_name(simplified)
    if carrying is not None:
        return carrying
    return resolved.read_as.envelope_json(name, simplified)


def _replaced(value: JSONValue, path: Loc, new: JSONValue) -> JSONValue:
    """`value` with the item at `path` replaced by `new`. `path` must come from a check of `value`."""
    if len(path) == 0:
        return new
    step, rest = path[0], path[1:]
    if isinstance(step, str) and isinstance(value, Mapping):
        return {**value, step: _replaced(value[step], rest, new)}
    if isinstance(step, int) and isinstance(value, tuple):
        return (*value[:step], _replaced(value[step], rest, new), *value[step + 1 :])
    msg = f"{path!r} does not lead into {value!r}"
    raise TypeError(msg)


__all__ = [
    "AcceptedField",
    "C",
    "Context",
    "Definition",
    "EmptyConfiguration",
    "Nested",
    "Problems",
    "RefusedField",
    "ResolvedField",
    "UnclaimedField",
    "WithFillValue",
    "as_kind",
    "asked",
    "canonical_fill_value",
    "canonical_of",
    "canonicalize",
    "configuration_of",
    "document_json",
    "field_key",
    "field_kind",
    "fields_of",
    "filed_under",
    "fill_value_as_written",
    "fill_value_problems",
    "format_of",
    "held",
    "is_field",
    "kind_of",
    "named_configuration",
    "no_rules",
    "own_key",
    "read_only",
    "resolve",
    "resolve_refined",
    "ruled",
    "spelled_canonically",
    "unchanged",
    "with_problems",
    "written_json",
]
