"""What every metadata entity can do for itself.

A codec, data type, chunk grid or chunk key encoding is one frozen
dataclass whose fields are its schema. Everything the layer knows about
a member's type is read off the field annotations by `_compile`: which
members there are, which may be absent, how each is type-checked, and
-- for a field typed as another entity -- that it is read through the
scope, written back as its own JSON, and put in canonical form by
recursing into it. Everything finer than a type -- a bound, a rule about
a member, members read together -- is the entity's own `__post_init__`,
which collects every problem it finds and raises once; `coerce` reports
those instead of raising.

`coerce` is the reading path: raw metadata in, the entity or the
reasons it is not one out, taking a `Context` -- the entities in scope
for this reading -- which the entities that contain other entities need.
`__init_subclass__` refuses, at class creation, every way of writing an
entity that would type-check and then misbehave somewhere that will not
name the class.

Composition -- what needs the document or the codec chain -- is the
entity's to answer too, through `incoming_problems`, `shape_problems`,
`fill_value_problems`, `transition` and `grid`, each taking the part of
the document it needs. The document that composes those answers is
`_document`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# Runtime imports, not `TYPE_CHECKING` ones: the string type aliases below
# (`TypeCheck`, `MemberTypes`) are resolved by `get_type_hints` at class
# creation, and a name that exists only for the type checker is a NameError
# then -- for this package and for any tool introspecting an entity.
from collections.abc import Callable, Mapping  # noqa: TC003
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, is_dataclass, replace
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Generic,
    Literal,
    NotRequired,
    TypeAlias,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import TypeIs, TypeVar, is_typeddict

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    MetadataValidationError,
    ValidationProblem,
)

if TYPE_CHECKING:
    from typing import Self

    from zarr_metadata.v3._parts import ArrayParts, ChunkGrid
    from zarr_metadata.v3._registry import Context

from zarr_metadata.v3._checks import (
    Loc,
    TypeCheck,
    as_tuples,
    is_bool,
    is_int,
    is_integer,
    is_json_value,
    is_metadata_field,
    is_str,
    named_configuration,
    one_of,
    problem,
    sequence_of,
    within,
)
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._compile import (
    FROM_NAME,
    MetadataFieldValue,
    check_for,
    declared_class_vars,
    element_annotations,
    field_hints,
    has_shape,
    is_class_var,
    is_from_name,
    is_metadata_field_type,
    is_nested_field,
    is_optional,
    is_union,
    own_annotations,
    shape_of,
    strip_annotation,
    type_check,
    unsubscripted,
)

EntityT = TypeVar("EntityT", bound="MetadataEntity")

JSONT_co = TypeVar(
    "JSONT_co", bound=ZarrV3MetadataFieldJSON, default=ZarrV3MetadataFieldJSON, covariant=True
)
"""What an entity's `to_json` returns: its own JSON type, named as the base's argument.

    class GzipCodec(CodecEntity[GzipCodecMetadata]): ...

Covariant, because it appears only in a return; defaulted, so a bare
`CodecEntity` -- in a field annotation, a table of entities, a scope --
means `CodecEntity[ZarrV3MetadataFieldJSON]` and admits every codec, each
of whose JSON types is assignable to that one (`ZarrV3NamedConfigJSON` is
`ReadOnly` and closed for exactly this). An entity that leaves it
defaulted is not wrong, only less informative.
"""


# A real alias, not a string one: entity modules subscript it as
# `Coerced[Self]` in a return annotation, and not all of them defer
# annotation evaluation.
Coerced: TypeAlias = tuple[EntityT | None, tuple[ValidationProblem, ...]]
"""The entity if it could be built, and every problem found.

One direction holds: no entity means at least one problem. The converse
does not -- a survivable problem, an unknown key, comes back *with* the
entity, because the entity is still readable and saying so is more
useful than refusing. A member of the wrong type is not survivable:
the entity's rules are written over a whole configuration, and an
entity is never built around a hole.

So test `entity is None` to decide whether to go on reading, and test the
problems to decide the verdict. They are different questions.
"""


ExtensionPointField = Literal[
    "data_type", "chunk_grid", "chunk_key_encoding", "codecs", "storage_transformers"
]
"""The v3 array metadata fields whose values name an extension.

Names are unique only within a point -- `bytes` is both a core codec and
a registered data type -- so every table in this package is keyed by
point and then by name, and an entity that contains other entities says
which point it reads them at.
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


def is_entity(value: object) -> TypeIs[MetadataEntity]:
    """`value` is an entity, of whatever JSON type.

    An `isinstance` against the generic base narrows an `object` to
    `MetadataEntity[Unknown]`; this narrows it to the defaulted
    `MetadataEntity`, whose `to_json` is any metadata field -- which is
    all that can be said of an entity met as an `object`.
    """
    return isinstance(value, MetadataEntity)


def _is_entity_kind(candidate: object) -> TypeIs[type[MetadataEntity]]:
    """`candidate` is an entity class; narrowed as `is_entity` narrows."""
    return isinstance(candidate, type) and issubclass(candidate, MetadataEntity)


def json_type_of(cls: type[MetadataEntity]) -> object:
    """The JSON type `cls` names for `to_json`; the default if it names none.

    Read off the subscripted base the class -- or the nearest ancestor
    that did -- was declared with, `CodecEntity[GzipCodecMetadata]`: the
    one place the correspondence between an entity and its public JSON
    type is written.
    """
    for klass in cls.__mro__:
        for base in klass.__dict__.get("__orig_bases__", ()):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, MetadataEntity):
                arguments = get_args(base)
                if len(arguments) == 1 and not isinstance(arguments[0], TypeVar):
                    return arguments[0]
    return ZarrV3MetadataFieldJSON


def contains_entity(annotation: object) -> bool:
    """Whether a value of this type holds a nested metadata field anywhere in it."""
    inner, _ = strip_annotation(annotation)
    if is_metadata_field_type(inner):
        return True
    origin = get_origin(inner)
    if is_union(inner):
        return any(contains_entity(arg) for arg in get_args(inner) if arg is not UNSET)
    if origin is tuple:
        return any(contains_entity(arg) for arg in get_args(inner) if arg is not Ellipsis)
    if is_typeddict(inner):
        return any(
            contains_entity(value) for value in get_type_hints(inner, include_extras=True).values()
        )
    if isinstance(inner, type) and is_dataclass(inner):
        return any(contains_entity(value) for value in field_hints(inner).values())
    return False


def _as_entity_kind(candidate: object) -> type[MetadataEntity] | None:
    """`candidate` as an entity type, or None if it is not one.

    In a function of its own so that the `isinstance`/`issubclass` pair
    narrows this parameter and not the caller's variable, which the
    caller goes on to read as the annotation it is.
    """
    candidate = unsubscripted(candidate)
    if _is_entity_kind(candidate):
        return candidate
    return None


def _entity_kinds(annotation: object) -> list[type[MetadataEntity]]:
    """Every entity type an annotation names, at any depth."""
    inner, _ = strip_annotation(annotation)
    kind = _as_entity_kind(inner)
    if kind is not None:
        return [kind]
    origin = get_origin(inner)
    arguments: tuple[object, ...] = get_args(inner)
    if is_union(inner):
        return [kind for arg in arguments if arg is not UNSET for kind in _entity_kinds(arg)]
    if origin is tuple:
        return [kind for arg in arguments if arg is not Ellipsis for kind in _entity_kinds(arg)]
    if isinstance(inner, type) and is_dataclass(inner) and not is_metadata_field_type(inner):
        return [kind for value in field_hints(inner).values() for kind in _entity_kinds(value)]
    return []


def _point_of(kind: type[MetadataEntity]) -> ExtensionPointField:
    """The point a nested entity kind is resolved at.

    Every nested field's kind has one by the time an entity exists --
    `__init_subclass__` refuses the class otherwise -- so this is the
    narrowing, not a second check.
    """
    point = kind.extension_point
    if point is None:
        msg = f"{kind.__name__} is registered at no single extension point"
        raise TypeError(msg)
    return point


def _fitting_branch(inner: object, value: object) -> object | None:
    """The branch of a union that holds an entity and whose shape `value` has."""
    for branch in get_args(inner):
        if branch is UNSET or not contains_entity(branch):
            continue
        if has_shape(shape_of(branch), value):
            return branch
    return None


def _resolve(
    annotation: object, value: object, context: Context, loc: Loc
) -> tuple[object, tuple[ValidationProblem, ...]]:
    """`value`, with every nested metadata field in it read as an entity in `context`.

    A field annotated with an entity type is resolved through the scope,
    at the point that kind of entity is registered at; an array of them
    element by element; a record holding one field by field. The value
    has passed its type check, so the shapes are the annotation's.
    """
    inner, _ = strip_annotation(annotation)
    if is_nested_field(inner):
        return context.coerce(_point_of(_entity_kinds(inner)[0]), value, loc)
    if is_union(inner):
        branch = _fitting_branch(inner, value)
        return (value, ()) if branch is None else _resolve(branch, value, context, loc)
    if get_origin(inner) is tuple:
        entries = cast("tuple[object, ...]", value)
        resolved: list[object] = []
        found: list[ValidationProblem] = []
        for position, (element, entry) in enumerate(
            zip(element_annotations(inner, len(entries)), entries, strict=True)
        ):
            item, problems = _resolve(element, entry, context, (*loc, position))
            resolved.append(item)
            found.extend(problems)
        return tuple(resolved), tuple(found)
    if isinstance(inner, type) and is_dataclass(inner) and not is_metadata_field_type(inner):
        entries = cast("Mapping[str, object]", value)
        members: dict[str, object] = {}
        found = []
        for name, field_annotation in field_hints(inner).items():
            if name not in entries:
                continue
            member, problems = _resolve(field_annotation, entries[name], context, (*loc, name))
            members[name] = member
            found.extend(problems)
        return inner(**members), tuple(found)
    return value, ()


def render_nested(annotation: object, value: object) -> object:
    """`value` as a document would write it: every nested entity in its JSON form."""
    if is_entity(value):
        return value.to_json()
    if isinstance(value, Opaque):
        return value.json
    inner, _ = strip_annotation(annotation)
    if is_union(inner):
        branch = _fitting_branch(inner, value)
        return value if branch is None else render_nested(branch, value)
    if get_origin(inner) is tuple:
        entries = cast("tuple[object, ...]", value)
        return tuple(
            render_nested(element, entry)
            for element, entry in zip(
                element_annotations(inner, len(entries)), entries, strict=True
            )
        )
    if isinstance(inner, type) and is_dataclass(inner) and not is_metadata_field_type(inner):
        return {
            name: render_nested(field_annotation, getattr(value, name))
            for name, field_annotation in field_hints(inner).items()
            if getattr(value, name) is not UNSET
        }
    return value


def canonicalize_nested(annotation: object, value: object) -> object:
    """`value` with every nested entity in its own canonical form."""
    if is_entity(value):
        return value.canonical()
    if isinstance(value, Opaque):
        return value
    inner, _ = strip_annotation(annotation)
    if is_union(inner):
        branch = _fitting_branch(inner, value)
        return value if branch is None else canonicalize_nested(branch, value)
    if get_origin(inner) is tuple:
        entries = cast("tuple[object, ...]", value)
        return tuple(
            canonicalize_nested(element, entry)
            for element, entry in zip(
                element_annotations(inner, len(entries)), entries, strict=True
            )
        )
    if (
        isinstance(inner, type)
        and is_dataclass(inner)
        and not is_metadata_field_type(inner)
        and is_dataclass(value)
        and not isinstance(value, type)
    ):
        return replace(
            value,
            **{
                name: canonicalize_nested(field_annotation, getattr(value, name))
                for name, field_annotation in field_hints(inner).items()
            },
        )
    return value


@dataclass(frozen=True, slots=True)
class Opaque(MetadataFieldValue):
    """A metadata field this reading did not turn into an entity.

    Carries the JSON the document wrote, so a reader holding a
    `CodecEntity | Opaque` has everything the document said in either
    case, and `reason` says which case it is. `out_of_scope` is a name no
    entity in this `Context` claims -- an extension this reader does not
    model, which is not an error and is the reader's cue to resolve it
    elsewhere. `invalid` is a name that *was* claimed and then refused,
    for the reasons reported alongside.
    """

    json: object
    reason: Literal["out_of_scope", "invalid"]


# The invariants, each a function of the compiled class returning why it
# is refused, or None. Every one names something that type-checks cleanly
# and then goes wrong somewhere that will not name the class.

_FINAL_ADVICE: Final[Mapping[str, str]] = {
    "canonical": (
        "which is the walk into contained entities; put the entity's own rewrite "
        "in `simplified`, which `canonical` calls after the walk"
    ),
}
"""What to do instead, for each method the base marks `@final`."""


def _members(cls: type[MetadataEntity]) -> dict[str, bool]:
    """The configuration members, each with whether it is required: the fields, less the envelope's."""
    return {
        name: not is_optional(annotation)
        for name, annotation in field_hints(cls).items()
        if not is_from_name(annotation)
    }


def _nested(cls: type[MetadataEntity]) -> dict[str, object]:
    """The fields that hold other entities, with their annotations."""
    return {
        name: annotation
        for name, annotation in field_hints(cls).items()
        if contains_entity(annotation)
    }


def _fields_are_json_shapes(cls: type[MetadataEntity]) -> str | None:
    hints = field_hints(cls)
    unread = sorted(
        name
        for name, annotation in hints.items()
        if not is_from_name(annotation) and check_for(annotation) is None
    )
    if len(unread) == 0:
        return None
    return (
        f"{cls.__name__}: "
        f"{'; '.join(f'{name} is annotated {hints[name]!r}' for name in unread)}"
        ", which is not a shape JSON takes. A field is int, float, bool, str, JSONValue, "
        "a Literal of names, tuple[T, ...] or tuple[T1, T2], a TypedDict or dataclass "
        "record, Mapping[str, V], a NewType, or an entity kind with Opaque "
        "(CodecEntity | Opaque); add | UNSET for an optional member, and put any finer "
        "rule in `__post_init__`"
    )


def _final_methods_are_not_overridden(cls: type[MetadataEntity]) -> str | None:
    # `@final` is a promise pyright checks in the author's editor; this
    # is the same promise for a class created without one.
    for name in vars(cls):
        if getattr(getattr(MetadataEntity, name, None), "__final__", False):
            return f"{cls.__name__} overrides `{name}`, {_FINAL_ADVICE.get(name, 'which is final')}"
    return None


def _nested_kinds_have_a_point(cls: type[MetadataEntity]) -> str | None:
    # `MetadataEntity` itself is registered at no single point, so a
    # field typed as one could not be resolved through a scope.
    unplaced = sorted(
        name
        for name, annotation in _nested(cls).items()
        if any(kind.extension_point is None for kind in _entity_kinds(annotation))
    )
    if len(unplaced) == 0:
        return None
    return (
        f"{cls.__name__}: the entity kind of {', '.join(unplaced)} has no "
        "`extension_point`; annotate it with `CodecEntity`, `DataTypeEntity` "
        "or `ChunkGridEntity`"
    )


def _entity_unions_lacking_opaque(annotation: object) -> bool:
    """Whether an entity kind appears in `annotation` without `Opaque` beside it."""
    inner, _ = strip_annotation(annotation)
    if is_union(inner):
        parts = [part for part in get_args(inner) if part is not UNSET]
        if any(is_metadata_field_type(part) and part is not Opaque for part in parts):
            return Opaque not in parts
        return any(_entity_unions_lacking_opaque(part) for part in parts)
    if is_metadata_field_type(inner):
        return inner is not Opaque
    if get_origin(inner) is tuple:
        return any(
            _entity_unions_lacking_opaque(part) for part in get_args(inner) if part is not Ellipsis
        )
    if isinstance(inner, type) and is_dataclass(inner):
        return any(_entity_unions_lacking_opaque(value) for value in field_hints(inner).values())
    return False


def _nested_fields_admit_opaque(cls: type[MetadataEntity]) -> str | None:
    # A nested field holds an `Opaque` when the name is out of scope, so
    # an annotation that excludes it lies to the type checker: reading
    # `codec.inner.level` would be accepted and then raise.
    lacking = sorted(
        name
        for name, annotation in _nested(cls).items()
        if _entity_unions_lacking_opaque(annotation)
    )
    if len(lacking) == 0:
        return None
    return (
        f"{cls.__name__}: {', '.join(lacking)} holds an entity but does not admit Opaque, "
        "which is what it holds when the name is out of scope; annotate it as the entity "
        "kind | Opaque"
    )


def _fields_do_not_shadow_class_variables(cls: type[MetadataEntity]) -> str | None:
    # A field of that name would go into the configuration and into the
    # JSON -- while the class variable it
    # shadows is what every other part of this layer reads.
    annotated = declared_class_vars(cls)
    shadowed = [
        name
        for name, annotation in own_annotations(cls).items()
        if name in annotated and annotated[name] is not cls and not is_class_var(annotation)
    ]
    if len(shadowed) == 0:
        return None
    return (
        f"{cls.__name__} declares {', '.join(shadowed)} as a field, shadowing a class "
        "variable of the same name; rename the field, or set the class variable instead"
    )


def _owed_class_variables_are_declared(cls: type[MetadataEntity]) -> str | None:
    # A class variable annotated with no value anywhere in the ancestry
    # is one the concrete entity owes: `identifier` for all of them,
    # `kind` for a codec, `bounds` for an integer type. Derived rather
    # than listed, so adding one to a family cannot forget to require it.
    annotated = declared_class_vars(cls)
    missing = sorted(name for name in annotated if not hasattr(cls, name))
    if len(missing) == 0:
        return None
    owed = ", ".join(f"{name} (annotated by {annotated[name].__name__})" for name in missing)
    return (
        f"{cls.__name__} does not declare {owed}; set each as a class variable, "
        "or pass base=True if this class exists only to be subclassed"
    )


def _is_name_type(part: object) -> bool:
    """A JSON type for the bare-name spelling: `str`, a `Literal` of names, or a `NewType` of `str`."""
    return (
        part is str
        or (get_origin(part) is Literal and all(isinstance(v, str) for v in get_args(part)))
        or getattr(part, "__supertype__", None) is str
    )


def _unaccepted(cls: type[MetadataEntity], name_type: object) -> list[str]:
    """The names a `Literal` name type lists that the entity does not accept."""
    if get_origin(name_type) is not Literal:
        return []
    return [value for value in get_args(name_type) if not cls.accepts(value)]


def _named_json_type_matches_what_is_written(cls: type[MetadataEntity]) -> str | None:
    # The named type is a promise about what `to_json` writes, held to
    # key by key: the spellings it admits are the ones the members make
    # the entity write, its names are ones the entity accepts, and its
    # configuration keys are the members. Value types are not compared
    # -- a nested entity's JSON type and its field type are different
    # spellings of one thing -- so that much stays with the tests.
    json_type = json_type_of(cls)
    if json_type is ZarrV3MetadataFieldJSON:
        return None
    parts = get_args(json_type) if is_union(json_type) else (json_type,)
    objects = [part for part in parts if is_typeddict(part)]
    names = [part for part in parts if _is_name_type(part)]
    if len(objects) > 1 or len(names) > 1 or len(objects) + len(names) != len(parts):
        return (
            f"{cls.__name__} names {json_type!r} as its JSON type, which is not an object "
            "TypedDict, a name type, or a union of one of each"
        )
    members = _members(cls)
    writes_bare = not any(members.values()) and cls.must_understand
    writes_object = len(members) != 0 or not cls.must_understand
    found: list[str] = []
    if writes_bare and len(names) == 0:
        found.append("lacks the bare name the entity writes when every member is absent")
    if not writes_bare and len(names) != 0:
        found.append("admits a bare name, which the entity never writes")
    if writes_object and len(objects) == 0:
        found.append("lacks the object the entity writes")
    if not writes_object and len(objects) != 0:
        found.append("admits an object, which the entity never writes")
    found.extend(
        f"lists the name(s) {', '.join(map(repr, unaccepted))}, which the entity does not accept"
        for name_type in names
        if len(unaccepted := _unaccepted(cls, name_type)) != 0
    )
    for obj in objects:
        try:
            resolved = get_type_hints(obj, include_extras=True)
        except NameError:
            # Declared inside a function under postponed annotations: the
            # names it uses are not reachable, so its keys go unjudged.
            continue
        hints = {key: strip_annotation(value)[0] for key, value in resolved.items()}
        # Requiredness from the resolved hints, not `__required_keys__`:
        # under postponed annotations a TypedDict's own metaclass cannot
        # see `NotRequired` inside a string.
        required = {key for key, value in resolved.items() if get_origin(value) is not NotRequired}
        extra = sorted(hints.keys() - {"name", "configuration", "must_understand"})
        if len(extra) != 0:
            found.append(f"has the key(s) {', '.join(extra)}, which no envelope has")
        if "name" not in hints:
            found.append("has no name key")
        elif len(unaccepted := _unaccepted(cls, hints["name"])) != 0:
            found.append(
                f"names {', '.join(map(repr, unaccepted))}, which the entity does not accept"
            )
        if not cls.must_understand and "must_understand" not in hints:
            found.append("has no must_understand key, which the entity writes")
        if len(members) == 0:
            if "configuration" in hints:
                found.append("has a configuration key, and the entity has no members")
            continue
        if "configuration" not in hints:
            found.append("has no configuration key, and the entity has members")
            continue
        if ("configuration" in required) != any(members.values()):
            found.append(
                "has configuration "
                + ("required" if "configuration" in required else "optional")
                + ", but a member is "
                + ("required" if any(members.values()) else "not required")
            )
        configuration = hints["configuration"]
        if not is_typeddict(configuration):
            continue
        try:
            configuration_hints = get_type_hints(configuration, include_extras=True)
        except NameError:
            continue
        keys = configuration_hints.keys()
        if set(keys) != set(members):
            found.append(
                f"has configuration keys {sorted(keys)!r} where the members are {sorted(members)!r}"
            )
            continue
        required_keys = {
            key
            for key, value in configuration_hints.items()
            if get_origin(value) is not NotRequired
        }
        misstated = sorted(
            key
            for key, member_required in members.items()
            if (key in required_keys) != member_required
        )
        if len(misstated) != 0:
            found.append(
                f"states {', '.join(misstated)} with a requiredness the field does not give it"
            )
    if len(found) == 0:
        return None
    return f"{cls.__name__} names {json_type!r} as its JSON type, which " + "; ".join(found)


def _codecs_are_of_a_kind(cls: type[MetadataEntity]) -> str | None:
    # The kind is the base class, and what a kind must answer is abstract
    # on it; a codec that skips the kind classes skips that.
    if not issubclass(cls, CodecEntity):
        return None
    if issubclass(cls, (ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec)):
        return None
    return (
        f"{cls.__name__} subclasses CodecEntity directly; subclass ArrayArrayCodec, "
        "ArrayBytesCodec or BytesBytesCodec, which says what the codec does to the array"
    )


def _class_variables_hold_listed_values(cls: type[MetadataEntity]) -> str | None:
    # Other entities' rules read these and have nothing to say about a
    # value outside the listed ones: the endian rule would fall silent.
    listed: tuple[tuple[str, tuple[object, ...]], ...] = ()
    if issubclass(cls, CodecEntity):
        listed = (("kind", get_args(CodecKind)),)
    elif issubclass(cls, DataTypeEntity):
        listed = (("scalar_storage", get_args(StorageClass)),)
    for name, values in listed:
        if hasattr(cls, name) and getattr(cls, name) not in values:
            return f"{cls.__name__} sets {name} = {getattr(cls, name)!r}, which is not one of {values!r}"
    return None


def _declared_defaults(cls: type[MetadataEntity]) -> dict[str, object]:
    """Each member's declared default, or `MISSING`.

    `@dataclass` has not run yet -- `__init_subclass__` runs first -- so
    a member declared with `field(...)` is still a `Field` here and its
    default has to be unwrapped.
    """
    defaulted: dict[str, object] = {}
    for key in _members(cls):
        declared: object = getattr(cls, key, MISSING)
        if type(declared) is Field:
            spec = cast("Field[object]", declared)
            declared = (
                MISSING
                if spec.default is MISSING and spec.default_factory is MISSING
                else spec.default
            )
        defaulted[key] = declared
    return defaulted


def _optional_members_default_to_unset(cls: type[MetadataEntity]) -> str | None:
    # Or `configuration` emits the member for every instance, so the
    # bare-name spelling becomes unreachable and a document gains a
    # member it never wrote.
    defaulted = _declared_defaults(cls)
    invented = [
        key
        for key, required in _members(cls).items()
        if not required and defaulted[key] is not UNSET
    ]
    if len(invented) == 0:
        return None
    return (
        f"{cls.__name__} gives the optional member(s) {', '.join(invented)} a default "
        "other than UNSET; write `| UNSET = UNSET` and read the meaning of absence where "
        "the member is used -- a default written into every document is not a member the "
        "document left out"
    )


def _required_members_have_no_default(cls: type[MetadataEntity]) -> str | None:
    # A required member with a default is an entity that can be built
    # without it -- and then serializes a document nobody wrote. A
    # conventional starting point is a `create_default` classmethod,
    # named so that asking for one is deliberate.
    defaulted = _declared_defaults(cls)
    presumed = [
        key for key, required in _members(cls).items() if required and defaulted[key] is not MISSING
    ]
    if len(presumed) == 0:
        return None
    return (
        f"{cls.__name__} gives the required member(s) {', '.join(presumed)} a default; "
        "either drop the default, or make the member optional with `| UNSET = UNSET`"
    )


_INVARIANTS: Final[tuple[Callable[[type[MetadataEntity]], str | None], ...]] = (
    _fields_are_json_shapes,
    _final_methods_are_not_overridden,
    _nested_kinds_have_a_point,
    _nested_fields_admit_opaque,
    _fields_do_not_shadow_class_variables,
    _owed_class_variables_are_declared,
    _named_json_type_matches_what_is_written,
    _codecs_are_of_a_kind,
    _class_variables_hold_listed_values,
    _optional_members_default_to_unset,
    _required_members_have_no_default,
)
"""What a compiled entity must satisfy, asked in this order at class creation."""


@dataclass(frozen=True)
class MetadataEntity(MetadataFieldValue, ABC, Generic[JSONT_co]):
    """One named entity, coerced from its metadata.

    Subclasses add their configuration members as fields, which is what
    makes them well-typed when read: `coerce` builds one only from
    metadata it accepted. Built by hand, the types are the caller's
    promise -- `__post_init__` judges values, not types. An optional member is
    typed `| UNSET` with a default of `UNSET`, so absence is representable
    -- and distinct from a `null` the document wrote -- and a canonical
    spelling can leave it out.

    Frozen, so an entity of hashable members is hashable. One holding a
    value out of scope is not, because that value is the JSON the document
    wrote and a JSON object is a `dict` -- the same way any frozen
    dataclass holding a list is unhashable. It cannot be an immutable
    mapping instead: `MappingProxyType` is unhashable too, and anything
    else stops `json.dumps` from serializing what `to_json` returns.

    A subclass writes its fields and, where the spec has something to
    say beyond their types, a `__post_init__` that collects every problem
    and raises `MetadataValidationError` once -- so `BloscCodec(clevel=99)`
    raises, and `coerce` reports the same problems instead. `coerce`,
    `configuration`, `to_json` and `canonical` are written once here
    against what the fields say, read off them as needed.
    """

    extension_point: ClassVar[ExtensionPointField | None] = None
    """Where this kind of entity is registered, if it is registered at one point.

    Set by `CodecEntity`, `DataTypeEntity` and `ChunkGridEntity`. It is
    what makes a field typed as one of those resolvable: the scope is
    asked at that point. `MetadataEntity` itself is the kind of the two
    points that take any entity, so it names none, and a field typed as
    bare `MetadataEntity` is refused at class creation.
    """

    must_understand: ClassVar[bool] = True
    """Whether a reader must understand this entity to read the array.

    `True`, the default and every codec: a reader that does not know it
    may not skip it.

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

    def __init_subclass__(cls, *, base: bool = False, **kwargs: object) -> None:
        """Compile the entity from its fields, and refuse one this layer cannot use.

        Every invariant in `_INVARIANTS` is asked, the first of which
        refuses a field annotation outside the shapes the compiler reads. Each names
        something that type-checks cleanly and then goes wrong later,
        somewhere that will not name this class; an import-time error in
        the extension's own module is the one place the author is looking.

        `base=True` for a class that exists to add a class variable
        rather than to be an entity -- `CodecEntity`, `IntegerDataType`.
        """
        super().__init_subclass__(**kwargs)
        if base:
            return
        if "__dataclass_fields__" in vars(cls):
            # `@dataclass(slots=True)` builds the class a second time from
            # the first one's dict: verified already, and its members are
            # slot descriptors now rather than the defaults the checks read.
            return
        for invariant in _INVARIANTS:
            message = invariant(cls)
            if message is not None:
                raise TypeError(message)

    @classmethod
    def accepts(cls, name: str) -> bool:
        """Whether `name` denotes this entity.

        Constant for all but the raw-bytes family, where one class covers
        every `r<N>`.
        """
        return name == cls.identifier

    @classmethod
    def coerce(cls, value: object, context: Context) -> Coerced[Self]:
        """`value` as this entity, or the reasons it is not one.

        `context` is the scope this reading is happening in; most entities
        have no use for it and ignore it.
        """
        name, configuration, _ = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), f"expected the {cls.identifier!r} entity")
        hints = field_hints(cls)
        if configuration is None:
            if any(_members(cls).values()):
                return None, problem(
                    ("configuration",),
                    f"{cls.identifier!r} requires a configuration",
                    "missing_key",
                )
            configuration = cast("Mapping[str, object]", {})
        members: dict[str, object] = {}
        found: list[ValidationProblem] = []
        for key in configuration:
            if key not in hints or is_from_name(hints[key]):
                found.extend(
                    problem(("configuration", key), f"unexpected key {key!r}", "unknown_key")
                )
        for key, annotation in hints.items():
            if is_from_name(annotation):
                members[key] = name
                continue
            if key not in configuration:
                if not is_optional(annotation):
                    found.extend(
                        problem(
                            ("configuration", key), f"missing required key {key!r}", "missing_key"
                        )
                    )
                continue
            # Normalized before the check, so a check only ever sees the
            # tuples the TypedDicts declare, never the lists raw JSON
            # arrives as. A member of the wrong type is reported and left
            # out; an unknown key inside it is survivable.
            member = as_tuples(configuration[key])
            problems = type_check(annotation)(member, ("configuration", key))
            found.extend(problems)
            if all(entry.kind == "unknown_key" for entry in problems):
                members[key] = member
        own = tuple(found)
        # A member that is itself an entity is read in the scope whatever
        # else was found: its problems are determinable, so they are
        # reported in the same pass.
        for key, annotation in hints.items():
            if key in members and contains_entity(annotation):
                members[key], nested = _resolve(
                    annotation, members[key], context, ("configuration", key)
                )
                found.extend(nested)
        found_all = tuple(found)
        if any(entry.kind != "unknown_key" for entry in own):
            # One of this entity's own members could not be read. That
            # leaves a hole, and the rules are written over a whole
            # configuration -- blosc's `typesize` requirement reads
            # `shuffle` -- so judging around it would be guessing: the
            # entity is not built, and the type problems stand alone.
            return None, found_all
        try:
            entity = cls(**members)
        except MetadataValidationError as refused:
            # `__post_init__` found values the spec disallows: reported
            # rather than raised, located under the configuration.
            return None, (*found_all, *within((), refused.problems))
        if any(entry.kind != "unknown_key" for entry in found_all):
            # A contained entity could not be read. This entity's own
            # rules ran -- an invalid inner is an `Opaque`, as an
            # out-of-scope one is -- but what is handed back is not an
            # entity that would be asked composition questions it cannot
            # answer.
            return None, found_all
        return entity, found_all

    @final
    def canonical(self) -> Self:
        """This entity in the simplest form that means the same thing.

        A *transformation*, asked for by `canonicalize_array_metadata_v3`
        and by nothing else. `to_json` does not apply it, because writing
        a document back is not the same as asking for it to be rewritten:
        a reader that reads and writes should not change bytes it was not
        asked to change.

        Two steps, each with one owner. Every contained entity is put in
        its own canonical form by walking the fields that hold one, which
        is read off the annotations and is this method's alone -- an
        override of it is refused at class creation. Then `simplified`,
        the entity's own rewrite, which is where an entity says that two
        spellings of its own members mean the same.
        """
        nested = _nested(type(self))
        walked = (
            self
            if len(nested) == 0
            else replace(
                self,
                **{
                    name: canonicalize_nested(annotation, getattr(self, name))
                    for name, annotation in nested.items()
                },
            )
        )
        return walked.simplified()

    def simplified(self) -> Self:
        """This entity with its own members in their simplest equivalent spelling.

        The hook `canonical` calls once every contained entity is in
        canonical form. Override it where two spellings of the entity's
        *own* members mean the same -- a rectilinear dimension's
        run-length encoding, a `typesize` that `noshuffle` ignores -- and
        return the entity rewritten. The default is the identity, and
        there is nothing to call `super()` for: the walk into contained
        entities is not this method's to keep.
        """
        return self

    def configuration(self) -> dict[str, object]:
        """This entity's configuration, as the document would write it.

        Faithful to every member the entity holds: `to_json` is
        serialization, not canonicalization, so nothing is simplified
        here. A contained entity is rendered as its own `to_json`, by
        walking the fields that hold one.

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
        members = self._configuration_members()
        for name, annotation in _nested(type(self)).items():
            if name in members:
                members[name] = render_nested(annotation, members[name])
        return deepcopy(members)

    def _configuration_members(self) -> dict[str, object]:
        """The members a configuration object would spell out.

        Every field but one the envelope carries some other way -- the
        `r<N>` width, which lives in the name.
        """
        return {
            key: value for key in _members(type(self)) if (value := getattr(self, key)) is not UNSET
        }

    def to_json(self) -> JSONT_co:
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

        The return type is the entity's own JSON type, named as the
        base's argument -- `GzipCodecObject`, `BytesCodecObject |
        BytesCodecName` -- and the `cast` below is the one place the
        package asserts that the dict it builds has that shape. Asserted
        rather than proven because a TypedDict cannot be built member by
        member from `dict[str, object]`; held to, twice: `__init_subclass__`
        refuses a named type whose shape disagrees with whether this
        entity ever writes a bare name or an object, and
        `tests/v3/test_entities.py` compiles the named type with
        `check_for` and runs every entity's output through it.
        """
        configuration = self.configuration()
        if len(configuration) == 0 and type(self).must_understand:
            return cast("JSONT_co", type(self).identifier)
        entry: dict[str, object] = {"name": type(self).identifier}
        if len(configuration) != 0:
            entry["configuration"] = configuration
        if not type(self).must_understand:
            entry["must_understand"] = False
        return cast("JSONT_co", entry)


@dataclass(frozen=True)
class CodecEntity(MetadataEntity[JSONT_co], base=True):
    """An entity that occupies a position in the codec pipeline.

    Of one of three kinds, each a base class: `ArrayArrayCodec`,
    `ArrayBytesCodec`, `BytesBytesCodec`. The kind fixes where in the
    pipeline the codec may stand, and what it must answer.
    """

    extension_point: ClassVar[ExtensionPointField] = CODECS
    """Where a codec is registered, and so where a field typed as one is resolved."""

    kind: ClassVar[CodecKind]
    """Set by the kind class."""

    variable_size: ClassVar[bool] = False
    """Whether this codec's output size depends on the bytes it is given.

    A compressor's does, so a shard index encoded with one has no size
    derivable from metadata alone, and the shard cannot be read.
    """

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """Why this codec cannot be applied to the array that reaches it.

        `incoming` is None once the chain can no longer say what reaches
        here, and the default answer to that is nothing: declining beats
        guessing. Locations are relative to this codec's `configuration`;
        an empty one lands on the codec itself.
        """
        return ()


@dataclass(frozen=True)
class ArrayArrayCodec(CodecEntity[JSONT_co], base=True):
    """A codec that transforms the array: what reaches the next codec is its to say."""

    kind: ClassVar[CodecKind] = "array_array"

    @abstractmethod
    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """What the next codec in the chain sees.

        `incoming` itself if this codec leaves the array's shape, grid and
        data type alone; the parts it hands on if it changes one; None if
        that cannot be determined from the metadata, which ends the
        judgments downstream rather than inventing them.
        """


@dataclass(frozen=True)
class ArrayBytesCodec(CodecEntity[JSONT_co], base=True):
    """The one codec in a pipeline that turns the array into bytes."""

    kind: ClassVar[CodecKind] = "array_bytes"


@dataclass(frozen=True)
class BytesBytesCodec(CodecEntity[JSONT_co], base=True):
    """A codec that transforms bytes, after the array is gone."""

    kind: ClassVar[CodecKind] = "bytes_bytes"


@dataclass(frozen=True)
class ChunkGridEntity(MetadataEntity[JSONT_co], base=True):
    """An entity that divides an array into the parts a pipeline encodes."""

    extension_point: ClassVar[ExtensionPointField] = CHUNK_GRID

    def shape_problems(self, array_shape: object) -> tuple[ValidationProblem, ...]:
        """Why this grid does not divide an array of `array_shape`.

        Locations are relative to the grid's `configuration`. Default:
        nothing, for a grid this package reads but has no such rule for.
        """
        return ()

    @abstractmethod
    def grid(self, array_shape: object) -> ChunkGrid:
        """What this grid divides an array of `array_shape` into.

        The array shape is a parameter because neither determines a grid
        alone: a grid whose own metadata cannot be read still has the
        array's rank, and rank is enough for several rules.
        """


@dataclass(frozen=True)
class DataTypeEntity(MetadataEntity[JSONT_co], base=True):
    """An entity that says how the array's scalars are stored.

    Only data types answer that, and every rule that turns on it -- a
    `bytes` codec is pointless before a single-byte type, a struct field
    cannot be variable-length -- asks a data type rather than consulting
    a table of names.
    """

    extension_point: ClassVar[ExtensionPointField] = DATA_TYPE

    scalar_storage: ClassVar[StorageClass]

    def storage_class(self) -> StorageClass | None:
        """How one scalar occupies bytes, or None if undetermined.

        None only for a composite whose parts are not all in scope: an
        answer would be a guess, and the rules that ask decline instead.
        """
        return type(self).scalar_storage

    @abstractmethod
    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        """Why `value` is not a fill value of this type, if it is not.

        Every data type answers this; one that accepts any fill value
        says so with `return ()`.
        """


__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "DATA_TYPE",
    "FROM_NAME",
    "STORAGE_TRANSFORMERS",
    "ArrayArrayCodec",
    "ArrayBytesCodec",
    "BytesBytesCodec",
    "ChunkGridEntity",
    "CodecEntity",
    "CodecKind",
    "Coerced",
    "DataTypeEntity",
    "ExtensionPointField",
    "JSONT_co",
    "Loc",
    "MetadataEntity",
    "Opaque",
    "StorageClass",
    "TypeCheck",
    "is_bool",
    "is_entity",
    "is_int",
    "is_integer",
    "is_json_value",
    "is_metadata_field",
    "is_str",
    "json_type_of",
    "named_configuration",
    "one_of",
    "problem",
    "sequence_of",
    "within",
]
