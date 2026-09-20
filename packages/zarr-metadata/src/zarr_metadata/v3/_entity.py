"""What every metadata entity can do for itself.

A codec, data type, chunk grid or chunk key encoding is three things at
once: a JSON shape, a set of constraints on the values in that shape, and
a canonical spelling. Keeping the three apart put one entity's knowledge
in four modules and needed a table per axis plus a drift test per table to
hold them together. Here they are one class per entity, and the class is
where methods bind:

- `coerce` is **type-space**: raw metadata in, the entity or the reasons
  it is not that entity out.
- `problems` is **value-space**: the entity is well-typed by construction,
  so this only asks whether its values are in range.
- `to_json` is **canonical**: the simplest spelling meaning the same, typed
  as the entity's own object TypedDict.

The TypedDicts stay: they model the JSON form, and the correspondence is
exact in both directions. A configuration TypedDict unpacked is the
dataclass constructor's signature, and the object TypedDict is what
`to_json` returns. `tests/v3/test_entities.py` asserts the first, so the
two cannot drift.

Everything that needs the document or the codec chain stays outside, in
`zarr_metadata.rules`, because an entity cannot answer it alone.

`coerce` takes a `Context`: the entities in scope for this reading. Most
entities ignore it -- a `gzip` codec is a `gzip` codec whatever else is
registered -- but the ones whose configuration contains other entities do
not. A `struct` data type holds field data types and a `sharding_indexed`
codec holds two codec pipelines, and neither can coerce its own
configuration without knowing what names are in scope inside it.

The shared plumbing lives here too: the member checks every entity needs
and the walk over a configuration that applies them. What stays with the
entity is the table saying which members it has -- that is the part that
is about blosc rather than about entities.
"""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, fields
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import ReadOnly, TypeIs

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    MetadataValidationError,
    ValidationProblem,
    is_json,
)
from zarr_metadata.v3._parts import ChunkGrid

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Self

    from zarr_metadata.model._validation import ProblemKind
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3._parts import ArrayParts
    from zarr_metadata.v3._registry import Context

EntityT = TypeVar("EntityT", bound="MetadataEntity")

# A real alias, not a string one: entity modules subscript it as
# `Coerced[Self]` in a return annotation, and not all of them defer
# annotation evaluation.
Coerced: TypeAlias = tuple[EntityT | None, tuple[ValidationProblem, ...]]
"""The entity if it could be built, and every problem found.

One direction holds: no entity means at least one problem. The converse
does not -- a survivable problem (an unknown key, an optional member of
the wrong type) comes back *with* the entity, because the entity is
still readable and saying so is more useful than refusing.

So test `entity is None` to decide whether to go on reading, and test the
problems to decide the verdict. They are different questions.
"""

Loc: TypeAlias = "tuple[str | int, ...]"

ExtensionPointField = Literal[
    "data_type", "chunk_grid", "chunk_key_encoding", "codecs", "storage_transformers"
]
"""The v3 array metadata fields whose values name an extension.

Here rather than in `_extension_points` because an entity that contains
other entities has to say which point it is reading them at, and
`_extension_points` also folds `r<N>` names -- which means importing the
data types, which import this.
"""

# Left to infer their `Literal` types rather than widened to
# `ExtensionPointField`: `Context.coerce` overloads on the field, so a
# call written with one of these constants gets the entity type back
# rather than the base. They are still assignable to the alias.
DATA_TYPE: Final = "data_type"
CHUNK_GRID: Final = "chunk_grid"
CHUNK_KEY_ENCODING: Final = "chunk_key_encoding"
CODECS: Final = "codecs"
STORAGE_TRANSFORMERS: Final = "storage_transformers"

StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How one scalar of a data type occupies bytes.

`single_byte` and `multi_byte` are both fixed-size; they differ only in
whether a byte order applies, which is what the `bytes` codec's `endian`
member is about.
"""

CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""The three pipeline positions the v3 spec sorts codecs into.

Declared by each codec, which is why there is no table of it: a name
does not have a pipeline position, a codec does.
"""

TypeCheck: TypeAlias = "Callable[[object, Loc], tuple[ValidationProblem, ...]]"
"""Whether one value has the type a member declares, and where if not."""

MemberTypes: TypeAlias = "Mapping[str, tuple[bool, TypeCheck]]"
"""Per configuration member: whether it is required, and its type check."""


def problem(
    loc: Loc, message: str, kind: ProblemKind = "invalid_type"
) -> tuple[ValidationProblem, ...]:
    """One problem, as the tuple every check returns."""
    return (ValidationProblem(loc, message, kind),)


def is_integer(value: object) -> TypeIs[int]:
    """A JSON integer: an `int`, and not a `bool`.

    `True` is an `int` in Python and `true` is not a number in JSON, so
    the two have to be told apart everywhere a number is expected.
    """
    return not isinstance(value, bool) and isinstance(value, int)


def is_int(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An integer, and not a bool -- JSON `true` is not the integer 1."""
    if not is_integer(value):
        return problem(loc, f"expected an integer, got {value!r}")
    return ()


def is_str(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, str):
        return problem(loc, f"expected a string, got {value!r}")
    return ()


def is_bool(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, bool):
        return problem(loc, f"expected a boolean, got {value!r}")
    return ()


def is_json_value(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """Any JSON value at all -- the widest type a member can declare."""
    if not is_json(value):
        return problem(loc, f"expected a JSON value, got {value!r}")
    return ()


def one_of(allowed: tuple[str, ...]) -> TypeCheck:
    """A member whose type is a closed set of names."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if value not in allowed:
            return problem(loc, f"expected one of {allowed!r}, got {value!r}", "invalid_value")
        return ()

    return check


def sequence_of(element: TypeCheck) -> TypeCheck:
    """A member whose type is a sequence, checked element by element."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, (list, tuple)):
            return problem(loc, f"expected a sequence, got {value!r}")
        elements: tuple[object, ...] = tuple(cast("list[object] | tuple[object, ...]", value))
        return tuple(
            found for index, entry in enumerate(elements) for found in element(entry, (*loc, index))
        )

    return check


def _as_tuples(value: object) -> object:
    """Every JSON array in `value`, at any depth, as a tuple.

    The TypedDicts spell a JSON array as a tuple throughout, so a member
    taken straight from parsed JSON would otherwise hold a list where its
    own type says tuple -- and two documents differing only in that would
    compare unequal.
    """
    if isinstance(value, (list, tuple)):
        entries = cast("list[object] | tuple[object, ...]", value)
        return tuple(_as_tuples(entry) for entry in entries)
    if isinstance(value, _Mapping):
        entries = cast("Mapping[str, object]", value)
        return {key: _as_tuples(entry) for key, entry in entries.items()}
    return value


def coerce_members(
    configuration: Mapping[str, object], types: MemberTypes
) -> tuple[dict[str, object], tuple[ValidationProblem, ...], frozenset[str]]:
    """The members `types` declares, taken from `configuration`.

    Returns what was accepted, every problem found, and the names of the
    required members that could not be read. Three kinds of problem, and
    they differ in that last part:

    - a key the entity does not declare says the value carries something
      extra, not that it is wrong;
    - an *optional* member of the wrong type leaves that member absent,
      and everything else about the entity is still readable -- a bad
      `index_location` says nothing about whether a shard's pipelines
      are well formed, and silencing them would lose a real judgment;
    - a *required* member missing or of the wrong type does stop it.
      There is no honest reading of a `blosc` whose level is a string.
    """
    problems: list[ValidationProblem] = []
    members: dict[str, object] = {}
    unreadable: set[str] = set()
    for key in configuration:
        if key not in types:
            problems.extend(
                problem(("configuration", key), f"unexpected key {key!r}", "unknown_key")
            )
    for key, (required, check) in types.items():
        if key not in configuration:
            if required:
                problems.extend(
                    problem(("configuration", key), f"missing required key {key!r}", "missing_key")
                )
                unreadable.add(key)
            continue
        # Normalized before the check, so a check only ever sees the tuples
        # the TypedDicts declare -- never the lists raw JSON arrives as.
        value = _as_tuples(configuration[key])
        found = check(value, ("configuration", key))
        problems.extend(found)
        # An unknown key says the value carries something extra, not that
        # it is the wrong type -- so the member is still readable, and
        # dropping it here would make `to_json` lose what was written.
        if all(entry.kind == "unknown_key" for entry in found):
            members[key] = value
        elif required:
            unreadable.add(key)
    return members, tuple(problems), frozenset(unreadable)


class ConfigurationType(Protocol):
    """What this layer reads off a configuration TypedDict.

    A Protocol rather than `type`, so that what is read off it is stated
    rather than assumed. Deliberately not `__required_keys__`: under
    `from __future__ import annotations` a TypedDict computes that from
    unresolved strings and reports every member required, which would
    make an extension's optional members silently mandatory. Requiredness
    is read from the resolved annotation instead.
    """

    __name__: str


def _unwrap(annotation: object) -> object:
    """An annotation without the qualifiers that are not its type.

    `NotRequired` and `Required` say whether a member must be present,
    which is the other half of a member table entry; `ReadOnly` says
    nothing about the value at all.
    """
    while get_origin(annotation) in (NotRequired, Required, ReadOnly):
        (annotation,) = get_args(annotation)
    return annotation


def check_for(annotation: object) -> TypeCheck | None:
    """The check an annotation implies, or None if it implies none.

    None for an annotation naming another structure -- a nested
    TypedDict, a recursive JSON alias, a tuple of either. Reading those
    off the annotation would be a TypedDict-to-checker compiler, which
    is a different package; the entity declares those itself.
    """
    annotation = _unwrap(annotation)
    if annotation is int:
        return is_int
    if annotation is bool:
        return is_bool
    if annotation is str:
        return is_str
    if annotation is JSONValue:
        return is_json_value
    if get_origin(annotation) is Literal:
        # Sorted, because the order `get_args` reports is not the order
        # the `Literal` was written in: two `Literal`s over the same
        # values compare and hash equal, so the first one built anywhere
        # in the process is the one every later one resolves to. The
        # check is a membership test either way; this is so the message
        # listing the values does not depend on import order.
        return one_of(tuple(sorted(cast("tuple[str, ...]", get_args(annotation)))))
    if get_origin(annotation) is tuple:
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            element = check_for(arguments[0])
            return None if element is None else sequence_of(element)
    return None


def is_required(annotation: object) -> bool:
    """Whether a configuration member must be present.

    From the resolved annotation rather than the TypedDict's
    `__required_keys__`, which is computed from unresolved strings and
    is wrong for a module using `from __future__ import annotations`.
    `ReadOnly` may wrap either way round, so it is peeled first.
    """
    while get_origin(annotation) is ReadOnly:
        (annotation,) = get_args(annotation)
    return get_origin(annotation) is not NotRequired


def derive_member_types(
    configuration: ConfigurationType,
) -> dict[str, tuple[bool, TypeCheck]]:
    """The member table a configuration TypedDict already describes.

    Requiredness is the TypedDict's, and so is the check wherever the
    annotation implies one. A member it does not imply one for is left
    out, for the entity to declare.
    """
    derived: dict[str, tuple[bool, TypeCheck]] = {}
    for member, annotation in get_type_hints(configuration, include_extras=True).items():
        check = check_for(annotation)
        if check is not None:
            derived[member] = (is_required(annotation), check)
    return derived


ValueRoutine: TypeAlias = "Callable[..., tuple[ValidationProblem, ...]]"
"""An entity's value-space judgment, over the members it was given."""


_MISSING_DEFAULT: Final = object()
"""Distinguishes "declared no default" from a default that is None or UNSET."""


def _no_value_problems(**members: object) -> tuple[ValidationProblem, ...]:
    """An entity whose types admit only valid values has nothing to add."""
    return ()


@dataclass(frozen=True, slots=True)
class Opaque:
    """A metadata field this reading did not turn into an entity.

    Carrying the JSON rather than dropping it is what makes the result a
    real union: `CodecEntity | Opaque` is exhaustive and narrows, where
    `CodecEntity | object` is just `object` and narrows to nothing.

    `reason` is the distinction a reader needs and could not otherwise
    make. `out_of_scope` is a name no entity in this `Context` claims --
    an extension this reader does not model, which is not an error and is
    the reader's cue to resolve it elsewhere. `invalid` is a name that
    *was* claimed and then refused; the reasons are in the problems
    reported alongside.
    """

    json: object
    reason: Literal["out_of_scope", "invalid"]


def _is_class_var(annotation: object) -> bool:
    """Whether an annotation says `ClassVar`.

    `from __future__ import annotations` leaves them as strings, so this
    reads the text when it gets one -- the same thing `dataclasses` does,
    and for the same reason: resolving the name needs a module namespace
    that is not available while the class is still being built.
    """
    if isinstance(annotation, str):
        stripped = annotation.strip()
        return stripped.startswith(("ClassVar[", "ClassVar", "typing.ClassVar"))
    return get_origin(annotation) is ClassVar


def _declared_class_vars(cls: type) -> dict[str, type]:
    """Every class variable annotated anywhere in `cls`'s ancestry.

    Mapped to the class that annotated it, so a message can say where the
    requirement comes from. Base first, so a redeclaration names the
    nearest ancestor.
    """
    found: dict[str, type] = {}
    for ancestor in reversed(cls.__mro__):
        for name, annotation in vars(ancestor).get("__annotations__", {}).items():
            if _is_class_var(annotation):
                found[name] = ancestor
    return found


# No `slots=True`, deliberately. It rebuilds the class, which on Python
# 3.11 and 3.12 leaves the zero-argument `super()` *in that same class's
# body* pointing at the class it replaced. Several entities call `super()`
# to narrow `to_json` and to adjust `configuration`, so they would each
# have to spell it `super(Cls, self)`. CPython fixed this in 3.13, so when
# that is the floor this is worth revisiting; the memory saved is small at
# document scale, which is why it has not been.
@dataclass(frozen=True)
class MetadataEntity:
    """One named entity, coerced from its metadata.

    Subclasses add their configuration members as fields, which is what
    makes them well-typed by construction: an instance exists only if
    `coerce` accepted the metadata that produced it. An optional member is
    typed `| None` with a default of `None`, so absence is representable
    and a canonical spelling can leave it out.

    Frozen, so an entity of hashable members is hashable. One holding a
    value out of scope is not, because that value is the JSON the document
    wrote and a JSON object is a `dict` -- the same way any frozen
    dataclass holding a list is unhashable. It cannot be an immutable
    mapping instead: `MappingProxyType` is unhashable too, and anything
    else stops `json.dumps` from serializing what `to_json` returns.

    Most subclasses declare `member_types` and nothing else: the default
    `coerce` and `to_json` are written once here against that table. The
    ones that override are the ones with something particular to say --
    a configuration containing other entities, a name that is a family
    rather than a constant, a member another member renders meaningless.
    """

    must_understand: ClassVar[bool] = True
    """Whether a reader that does not know this entity may skip it.

    A property of the *kind* of metadata, not of a use of it: a codec is
    something you must understand, every time it appears, because
    ignoring one gives wrong bytes. Consolidated metadata is the opposite
    and is unconditionally skippable. Neither is a per-occurrence choice,
    so neither is a configuration member -- which is why this is a class
    variable and not a field.

    The spec permits `must_understand: false` on a codec; this package
    treats that as an oversight and refuses it. Where the flag does earn
    its keep -- an unknown top-level extension field a reader really can
    skip -- it stays per-occurrence, on `ZarrV3NamedConfig`.
    """

    identifier: ClassVar[str]
    """The name this entity is registered under.

    Usually the `name` the metadata carries. The raw-bytes data types are
    the exception: every `r<N>` spelling is one family, so the family gets
    an invented identifier that no real name can collide with.
    """

    configuration_type: ClassVar[ConfigurationType | None] = None
    """The TypedDict describing this entity's `configuration` in JSON.

    None for an entity that has no configuration. Everything else about
    the members is read off it at class creation, so the JSON shape is
    stated once: `member_types` and `configuration_required` are both
    derived, and the constructor is held to the same keys by
    `tests/v3/test_entities.py`.
    """

    member_types: ClassVar[MemberTypes] = MappingProxyType({})
    """The configuration members, and the type each one takes.

    Derived from `configuration_type`. A class declares an entry here
    only for a member whose annotation names another structure -- a
    nested TypedDict, a recursive JSON alias -- which is where reading
    the check off the annotation would take a compiler.
    """

    configuration_required: ClassVar[bool] = False
    """Whether the bare-name spelling says too little for this entity.

    The spec permits a bare name "if no configuration metadata is
    required", so this is true exactly when some member is required --
    which the configuration TypedDict already says.
    """

    def __init_subclass__(cls, *, base: bool = False, **kwargs: object) -> None:
        """Refuse a subclass that is not an entity this layer can use.

        Every check here has the same shape: something that type-checks
        cleanly and then goes wrong later, somewhere that will not name
        this class. An import-time error in the extension's own module is
        the one place the author is looking.

        `base=True` for a class that exists to add a class variable
        rather than to be an entity -- `CodecEntity`, `IntegerDataType`.
        """
        super().__init_subclass__(**kwargs)
        if base:
            return
        if "problems" in cls.__dict__:
            # Value rules are `value_problems`, a static routine over the
            # members. An override named `problems` is a rule that would
            # never run, and nothing else would say so.
            msg = (
                f"{cls.__name__} defines `problems`; value rules belong in "
                "`value_problems`, which takes the members rather than an entity"
            )
            raise TypeError(msg)
        if "__post_init__" in cls.__dict__:
            # `coerce` builds through `unchecked`, which bypasses
            # `__init__` and so never reaches `__post_init__`. Rules put
            # there would hold for a hand-built entity and be silently
            # absent for every entity read from a document -- the one
            # direction that matters.
            msg = (
                f"{cls.__name__} defines `__post_init__`, which `unchecked` does "
                "not reach; value rules belong in `value_problems`"
            )
            raise TypeError(msg)
        if "configuration_required" in vars(cls):
            msg = (
                f"{cls.__name__} declares `configuration_required`, which follows "
                "from whether its configuration has a required member"
            )
            raise TypeError(msg)
        if cls.configuration_type is not None:
            # Before every guard below, because they read the table.
            declared = dict(vars(cls).get("member_types", {}))
            derived = derive_member_types(cls.configuration_type)
            undeclared = sorted(
                set(get_type_hints(cls.configuration_type)) - set(derived) - set(declared)
            )
            if len(undeclared) != 0:
                msg = (
                    f"{cls.__name__} declares no check for {', '.join(undeclared)}, "
                    "whose annotation does not imply one"
                )
                raise TypeError(msg)
            hints = get_type_hints(cls.configuration_type, include_extras=True)
            misstated = sorted(
                member
                for member, (required, _) in declared.items()
                if member in hints and required != is_required(hints[member])
            )
            if len(misstated) != 0:
                # The check is the entity's to write; whether the member
                # may be left out is the configuration's to say, and a
                # declared entry that disagrees is the drift this
                # derivation exists to rule out.
                msg = (
                    f"{cls.__name__} declares {', '.join(misstated)} with a requiredness "
                    "its configuration does not give it"
                )
                raise TypeError(msg)
            cls.member_types = {**derived, **declared}
            cls.configuration_required = any(required for required, _ in cls.member_types.values())
        annotated = _declared_class_vars(cls)
        shadowed = [
            name
            for name in vars(cls).get("__annotations__", {})
            if name in annotated
            and annotated[name] is not cls
            and not _is_class_var(vars(cls)["__annotations__"][name])
        ]
        if len(shadowed) != 0:
            # A field of that name would go into `member_types`, into the
            # configuration, and into the JSON -- while the class variable
            # it shadows is what every other part of this layer reads.
            msg = (
                f"{cls.__name__} declares {', '.join(shadowed)} as a field, "
                "shadowing a class variable of the same name"
            )
            raise TypeError(msg)
        # A class variable annotated with no value anywhere in the
        # ancestry is one the concrete entity owes: `identifier` for all
        # of them, `kind` for a codec, `bounds` for an integer type.
        # Derived rather than listed, so adding one to a family cannot
        # forget to require it.
        missing = [name for name in annotated if not hasattr(cls, name)]
        if len(missing) != 0:
            msg = f"{cls.__name__} does not declare {', '.join(sorted(missing))}"
            raise TypeError(msg)
        # A member's default decides whether the entity can exist without
        # it, so the two kinds have opposite rules. `@dataclass` has not
        # run yet, so a member declared with `field(...)` is still a
        # `Field` here and its default has to be unwrapped.
        defaulted: dict[str, object] = {}
        for key in cls.member_types:
            declared: object = getattr(cls, key, _MISSING_DEFAULT)
            if type(declared) is Field:
                # `field(...)`, so the default is inside it rather than
                # being the attribute. `@dataclass` has not unwrapped it
                # yet -- this hook runs first.
                spec = cast("Field[object]", declared)
                declared = (
                    _MISSING_DEFAULT
                    if spec.default is MISSING and spec.default_factory is MISSING
                    else spec.default
                )
            defaulted[key] = declared
        # An optional member defaults to UNSET or `configuration` emits it
        # for every instance, so the bare-name spelling becomes
        # unreachable and a document gains a member it never wrote.
        invented = [
            key
            for key, (required, _) in cls.member_types.items()
            if not required and defaulted[key] is not UNSET
        ]
        if len(invented) != 0:
            msg = (
                f"{cls.__name__} gives the optional member(s) "
                f"{', '.join(invented)} a default other than UNSET"
            )
            raise TypeError(msg)
        # A required member with a default is an entity that can be built
        # without it -- and then serializes a document nobody wrote. A
        # conventional starting point is a `create_default` classmethod,
        # named so that asking for one is deliberate.
        presumed = [
            key
            for key, (required, _) in cls.member_types.items()
            if required and defaulted[key] is not _MISSING_DEFAULT
        ]
        if len(presumed) != 0:
            msg = (
                f"{cls.__name__} gives the required member(s) "
                f"{', '.join(presumed)} a default; required members have none"
            )
            raise TypeError(msg)

    @classmethod
    def accepts(cls, name: str) -> bool:
        """Whether `name` denotes this entity.

        Constant for all but the raw-bytes family, where one class covers
        every `r<N>`.
        """
        return name == cls.identifier

    @classmethod
    def prepare(
        cls, members: dict[str, object], context: Context
    ) -> tuple[dict[str, object], tuple[ValidationProblem, ...]]:
        """The members, with any that are themselves entities read as such.

        The seam between `coerce_members`, which knows types, and
        `value_problems`, which knows values: a `struct` cannot ask
        whether a field is fixed-size until that field's data type is an
        entity. Default: nothing to convert.
        """
        return members, ()

    @classmethod
    def coerce(cls, value: object, context: Context) -> Coerced[Self]:
        """`value` as this entity, or the reasons it is not one.

        `context` is the scope this reading is happening in; most entities
        have no use for it and ignore it.
        """
        name, configuration, _ = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), f"expected the {cls.identifier!r} entity")
        if configuration is None:
            if cls.configuration_required:
                return None, problem(
                    ("configuration",),
                    f"{cls.identifier!r} requires a configuration",
                    "missing_key",
                )
            configuration = cast("Mapping[str, object]", {})
        members, found, unreadable = coerce_members(configuration, cls.member_types)
        if len(unreadable) == 0:
            # Before judging: a member that is itself an entity has to be
            # one before its container's value rules can ask it anything.
            members, nested = cls.prepare(members, context)
            found = (*found, *nested)
        if len(unreadable) != 0:
            # A member that could not be read leaves a hole, and the value
            # rules are written over a whole configuration -- blosc's
            # `typesize` requirement reads `shuffle`. Judging around the
            # hole would be guessing, so the type problems stand alone.
            return None, found
        found = (*found, *within((), cls.value_problems(**members)))
        if any(entry.kind != "unknown_key" for entry in found):
            return None, found
        # Already asked, so do not ask again on the way in.
        return cls.unchecked(**members), found

    def canonical(self) -> Self:
        """This entity in the simplest form that means the same thing.

        A *transformation*, asked for by `canonicalize_array_metadata_v3`
        and by nothing else. `to_json` does not apply it, because writing
        a document back is not the same as asking for it to be rewritten:
        a reader that reads and writes should not change bytes it was not
        asked to change.

        Default: entities are already canonical. Override where two
        spellings of a member mean the same -- a rectilinear dimension's
        run-length encoding, a `typesize` that `noshuffle` ignores -- and
        where a contained entity has its own canonical form.
        """
        return self

    def configuration(self) -> dict[str, object]:
        """This entity's configuration, as the document would write it.

        Faithful to every member the entity holds: `to_json` is
        serialization, not canonicalization, so nothing is simplified
        here. Override only to render a member that is not already JSON,
        such as a contained entity.

        Absent optional members are left out, which is what makes the
        bare-name spelling reachable. Absence is `UNSET`, never `None`:
        this package holds `None` to mean a JSON `null` the document
        actually wrote, and `scale_offset` is a real case where `null`
        and absent are different documents.

        Deep-copied, because a member can be an arbitrary JSON value: a
        `scale_offset` offset may be an object, and handing the caller
        the entity's own dict would let them mutate a frozen entity
        through the document it returned.
        """
        return deepcopy(self._configuration_members())

    value_problems: ClassVar[ValueRoutine] = staticmethod(_no_value_problems)
    """Every value among the members the spec disallows.

    A routine rather than a method, because judging values does not need
    an entity -- and needing one would mean an invalid one had been
    built. Each entity supplies its own, taking
    `Unpack[<Entity>Configuration]`: the same spelling the constructor
    takes, receiving only the members that are present.

    Typed loosely here because the base does not know any entity's
    configuration, and saying so is the truth. Call a specific routine by
    its own name to have the arguments checked.

    Locations are relative to the entity's `configuration`.
    """

    def __post_init__(self) -> None:
        """Refuse to exist with values the spec disallows.

        So an instance is the value guarantee, not just the type one:
        `BloscCodec(clevel=99)` raises rather than serializing a document
        no reader will accept. `coerce` asks `value_problems` first and
        reports, so reading a bad document still returns problems rather
        than raising, and `unchecked` is the door for a caller that has
        already asked.
        """
        found = type(self).value_problems(**self._members())
        if len(found) != 0:
            raise MetadataValidationError(found)

    def _members(self) -> dict[str, object]:
        """Every member this entity holds, unrendered.

        The dataclass's own fields, which is what `value_problems`
        judges: a member is a member whether or not the JSON spells it
        as a configuration key. The raw-bytes family is the case that
        separates the two -- its width lives in its name, so it has a
        field and no configuration at all.
        """
        return {
            field_.name: value
            for field_ in fields(self)
            if (value := getattr(self, field_.name)) is not UNSET
        }

    def _configuration_members(self) -> dict[str, object]:
        """The members a configuration object would spell out.

        `_members` minus anything the envelope carries some other way.
        """
        return {
            key: value
            for key in type(self).member_types
            if (value := getattr(self, key)) is not UNSET
        }

    @classmethod
    def unchecked(cls, **members: object) -> Self:
        """This entity, without asking whether its values are allowed.

        For a caller that has already asked -- `coerce` does, so that it
        can report the answer instead of raising it. Named so that
        choosing it is deliberate.

        Unchecked means *value*-unchecked. A member this entity does not
        declare, or one with neither a value nor a default, is still a
        `TypeError`: those produce an entity that cannot be repred,
        compared or hashed, which no caller is asking for.
        """
        declared = {field_.name: field_ for field_ in fields(cls)}
        unknown = sorted(members.keys() - declared.keys())
        if len(unknown) != 0:
            msg = f"{cls.__name__} has no member(s) {', '.join(unknown)}"
            raise TypeError(msg)
        entity = object.__new__(cls)
        for name, field_ in declared.items():
            if name in members:
                object.__setattr__(entity, name, members[name])
            elif field_.default is not MISSING:
                object.__setattr__(entity, name, field_.default)
            elif field_.default_factory is not MISSING:  # pragma: no cover - none today
                object.__setattr__(entity, name, field_.default_factory())
            else:
                # Leaving it unset would give an entity whose `repr`,
                # `==` and `hash` raise `AttributeError` on access.
                msg = f"{cls.__name__} is missing a value for {name!r}"
                raise TypeError(msg)
        return entity

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """This entity as a document would write it.

        Faithful to every member it models: read a document, write it
        back, and those come out as they went in. Ask `canonical` first
        if you want the simplest equivalent spelling.

        A member this entity does not model is not one of them. It is
        reported as `unknown_key` and not held, so writing back drops it
        -- which only a caller who took the problems as data and went on
        past that one can reach, because `from_json` raises on it. A
        caller who needs the bytes preserved has the JSON it passed in,
        and `Opaque` is where unmodelled metadata belongs.

        What is *not* preserved is the envelope's spelling, because the
        entity does not model it: a bare name, `{"name": x}`, and
        `{"name": x, "configuration": {}}` all mean the same and all read
        to the same entity, so all three write back as the bare name.
        `must_understand` follows the entity's own class variable, so it
        is omitted for everything this package models today.

        Subclasses narrow the return type to their own object TypedDict,
        which is the JSON form this dataclass models.
        """
        configuration = self.configuration()
        if len(configuration) == 0 and type(self).must_understand:
            return cast("ZarrV3MetadataFieldJSON", type(self).identifier)
        entry: dict[str, object] = {"name": type(self).identifier}
        if len(configuration) != 0:
            entry["configuration"] = configuration
        if not type(self).must_understand:
            entry["must_understand"] = False
        return cast("ZarrV3MetadataFieldJSON", entry)


@dataclass(frozen=True)
class CodecEntity(MetadataEntity, base=True):
    """An entity that occupies a position in the codec pipeline."""

    kind: ClassVar[CodecKind]

    variable_size: ClassVar[bool] = False
    """Whether this codec's output size depends on the bytes it is given.

    A compressor's does, so a shard index encoded with one has no size
    derivable from metadata alone, and the shard cannot be read.
    """

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """Why this codec cannot be applied to the array that reaches it.

        `incoming` is None once the chain can no longer say what reaches
        here, and the default answer to that is nothing: declining beats
        guessing. Locations are relative to this codec's `configuration`,
        as `problems`' are; an empty one lands on the codec itself.
        """
        return ()

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """What the next codec in the chain sees, or None if undeterminable.

        Only an array-to-array codec has anything to say: the two later
        kinds end shape propagation by construction, one by consuming the
        array and the other by never having had it.

        The default is None, so a modelled codec that forgets to say how
        it transforms the array stops propagation rather than silently
        claiming to leave it alone. Failing closed here costs a judgment;
        failing open would invent one.
        """
        return None


@dataclass(frozen=True)
class ChunkGridEntity(MetadataEntity, base=True):
    """An entity that divides an array into the parts a pipeline encodes."""

    def shape_problems(self, array_shape: object) -> tuple[ValidationProblem, ...]:
        """Why this grid does not divide an array of `array_shape`.

        Locations are relative to the grid's `configuration`. Default:
        nothing, for a grid this package reads but has no such rule for.
        """
        return ()

    def grid(self, array_shape: object) -> ChunkGrid:
        """What this grid divides an array of `array_shape` into.

        The array shape is a parameter because neither determines a grid
        alone: a grid whose own metadata cannot be read still has the
        array's rank, and rank is enough for several rules.
        """
        return ChunkGrid.unreadable(array_shape)


@dataclass(frozen=True)
class DataTypeEntity(MetadataEntity, base=True):
    """An entity that says how the array's scalars are stored.

    Only data types answer that, and every rule that turns on it -- a
    `bytes` codec is pointless before a single-byte type, a struct field
    cannot be variable-length -- asks a data type rather than consulting
    a table of names.
    """

    scalar_storage: ClassVar[StorageClass]

    def storage_class(self) -> StorageClass | None:
        """How one scalar occupies bytes, or None if undetermined.

        None only for a composite whose parts are not all in scope: an
        answer would be a guess, and the rules that ask decline instead.
        """
        return type(self).scalar_storage

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        """Why `value` is not a fill value of this type, if it is not.

        Default: nothing. A data type this package does not model accepts
        whatever its extension says it does, and guessing would reject
        valid documents.
        """
        return ()


def within(prefix: Loc, problems: Sequence[ValidationProblem]) -> tuple[ValidationProblem, ...]:
    """One entity's problems, located in the document that holds it.

    An entity reports relative to its own `configuration`, so that is what
    goes between the field and the member. A problem with an empty
    location is about the entity itself -- a malformed `r<N>` name, a
    codec that cannot encode what reaches it -- and lands on the field.
    """
    return tuple(
        ValidationProblem(
            (*prefix, *(("configuration", *found.loc) if len(found.loc) != 0 else ())),
            found.message,
            found.kind,
        )
        for found in problems
    )


def named_configuration(
    value: object,
) -> tuple[str | None, Mapping[str, object] | None, bool]:
    """Split metadata into `(name, configuration, must_understand)`.

    The shared shape every entity arrives in: a bare name, or an object
    carrying one. A `None` name means the value is not a metadata field at
    all; a `None` configuration means the bare spelling was used.
    """
    if isinstance(value, str):
        return value, None, True
    if not isinstance(value, _Mapping):
        return None, None, True
    entry = cast("Mapping[str, object]", value)
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, True
    configuration = entry.get("configuration")
    must_understand = entry.get("must_understand", True)
    return (
        name,
        cast("Mapping[str, object]", configuration)
        if isinstance(configuration, _Mapping)
        else None,
        must_understand if isinstance(must_understand, bool) else True,
    )


__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "DATA_TYPE",
    "STORAGE_TRANSFORMERS",
    "ChunkGridEntity",
    "CodecEntity",
    "CodecKind",
    "Coerced",
    "DataTypeEntity",
    "ExtensionPointField",
    "Loc",
    "MemberTypes",
    "MetadataEntity",
    "Opaque",
    "StorageClass",
    "TypeCheck",
    "ValueRoutine",
    "coerce_members",
    "is_bool",
    "is_int",
    "is_integer",
    "is_json_value",
    "is_str",
    "named_configuration",
    "one_of",
    "problem",
    "sequence_of",
    "within",
]
