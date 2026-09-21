"""What a field annotation says, read off it: the type, the bounds, the rules.

An entity's dataclass fields are its schema, and this module is the
compiler over them. `check_for` turns an annotation into its type check
-- a scalar, a `Literal`, arrays homogeneous or fixed, unions, a nested
object described by a TypedDict or a record dataclass -- through a
registry of shapes that `register_check` keeps open, so a shape this
module does not know can be taught to it from outside. The bound
vocabulary (`Ge`, `Le`, `Interval`, ...) rides in `Annotated` and
`value_check_for` compiles it, at whatever depth it sits. `validates`
marks a rule about one member. `derive_member_types` is what an entity
reads its member table off.

Nothing here knows what an entity is. A nested metadata field is a shape
like any other, registered by `_entity` through the same door, with the
JSON shape and the description the compiler needs of it.
"""

from __future__ import annotations

# Runtime imports, not `TYPE_CHECKING` ones: the string type aliases below
# (`TypeCheck`, `MemberTypes`) are resolved by `get_type_hints` at class
# creation, and a name that exists only for the type checker is a NameError
# then -- for this package and for any tool introspecting an entity.
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    NotRequired,
    Required,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import ReadOnly, is_typeddict

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.v3._checks import (
    is_bool,
    is_int,
    is_integer,
    is_json_value,
    is_str,
    one_of,
    problem,
    sequence_of,
)

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v3._checks import Loc, TypeCheck


class _FromName:
    """The marker behind `FROM_NAME`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "FROM_NAME"


FROM_NAME: Final = _FromName()
"""Marks a field carried by the metadata envelope's `name`, not its configuration.

    data_type_name: Annotated[str, FROM_NAME]

A member all the same -- `value_problems` judges it -- but not a
configuration key, so it is neither read from nor written to a
`configuration` object. The raw-bytes family is the case: `r<N>` keeps its
width in its name and has no configuration at all.
"""


@dataclass(frozen=True, slots=True)
class Ge:
    """`Annotated[int, Ge(1)]`: the value is at least `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Gt:
    """`Annotated[int, Gt(0)]`: the value is more than `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Le:
    """`Annotated[int, Le(9)]`: the value is at most `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Lt:
    """`Annotated[int, Lt(10)]`: the value is less than `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Interval:
    """`Annotated[int, Interval(ge=0, le=9)]`: the value lies within these bounds.

    These five are the `annotated_types` vocabulary -- what pydantic reads
    and msgspec's `Meta` mirrors -- so a reader recognises them. Defined
    here rather than imported, so the package keeps its one dependency.
    A bound is a value rule: it runs only once the member has the type it
    declared, at whatever depth the annotation puts it, so a bound on an
    array's element type judges each element at its own position.
    """

    ge: int | float | None = None
    gt: int | float | None = None
    le: int | float | None = None
    lt: int | float | None = None


def strip_annotation(annotation: object) -> tuple[object, tuple[object, ...]]:
    """An annotation's type, and the metadata `Annotated` wrapped it in.

    `NotRequired`, `Required` and `ReadOnly` are qualifiers on a TypedDict
    key, not part of the value's type; peeled with the `Annotated` layers,
    in whatever order they were written.
    """
    metadata: list[object] = []
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            inner, *extras = get_args(annotation)
            metadata.extend(extras)
            annotation = inner
        elif origin in (NotRequired, Required, ReadOnly):
            (annotation,) = get_args(annotation)
        else:
            return annotation, tuple(metadata)


def own_annotations(klass: type) -> dict[str, object]:
    """A class's own annotations, unevaluated.

    From 3.14 a class does not carry an `__annotations__` dict until it is
    asked for one, and asking evaluates every annotation at once -- so a
    `ClassVar` naming something imported only for the type checker would
    fail the whole class. `annotationlib` can hand them back as the text
    they were written as, which is what the callers here want anyway:
    class variables are skipped by text before anything is evaluated.
    Earlier versions leave the dict on the class, strings or values as
    the module chose.
    """
    if sys.version_info >= (3, 14):
        import annotationlib

        return dict(annotationlib.get_annotations(klass, format=annotationlib.Format.STRING))
    return dict(vars(klass).get("__annotations__", {}))


def field_hints(cls: type) -> dict[str, object]:
    """The dataclass fields of `cls`, resolved, base first.

    Each class's own annotations are resolved in that class's module,
    and class variables are skipped *before* resolving, by text -- so a
    `ClassVar` whose annotation names something imported only for the
    type checker cannot fail class creation. `@dataclass` sees the same
    set, in the same order.
    """
    hints: dict[str, object] = {}
    for ancestor in reversed(cls.__mro__):
        raw = {
            name: annotation
            for name, annotation in own_annotations(ancestor).items()
            if not is_class_var(annotation)
        }
        if len(raw) == 0:
            continue
        shell = type("_Fields", (), {"__annotations__": raw, "__module__": ancestor.__module__})
        hints.update(get_type_hints(shell, include_extras=True))
    return hints


def is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def is_optional(annotation: object) -> bool:
    """Whether a field may be absent: its type admits `UNSET`."""
    inner, _ = strip_annotation(annotation)
    return is_union(inner) and any(arg is UNSET for arg in get_args(inner))


def describe(annotation: object) -> str:
    """The annotation as a message would name it: "an integer", "an object"."""
    inner, _ = strip_annotation(annotation)
    for registration in _CHECK_COMPILERS:
        if registration.description is not None and registration.predicate(inner):
            return registration.description
    if inner is int:
        return "an integer"
    if inner is bool:
        return "a boolean"
    if inner is str:
        return "a string"
    if inner is JSONValue:
        return "a JSON value"
    origin = get_origin(inner)
    if origin is Literal:
        return f"one of {tuple(sorted(get_args(inner)))!r}"
    if is_union(inner):
        branches = [arg for arg in get_args(inner) if arg is not UNSET]
        return " or ".join(describe(branch) for branch in branches)
    if origin is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return f"an array of {describe(arguments[0])} elements"
        if len(arguments) == 2:
            return f"a [{describe(arguments[0])}, {describe(arguments[1])}] pair"
        return f"an array of {len(arguments)} elements"
    if is_typeddict(inner) or is_dataclass(inner):
        return "an object"
    return "a value"


def shape_of(annotation: object) -> str | None:
    """The top-level JSON shape an annotation admits, for choosing a union branch.

    None means any shape -- a JSON value, or a union that mixes them.
    """
    inner, _ = strip_annotation(annotation)
    for registration in _CHECK_COMPILERS:
        if registration.shape is not None and registration.predicate(inner):
            return registration.shape
    if inner is int:
        return "int"
    if inner is bool:
        return "bool"
    if inner is str:
        return "str"
    origin = get_origin(inner)
    if origin is Literal:
        values = get_args(inner)
        return "int" if all(isinstance(value, int) for value in values) else "str"
    if origin is tuple:
        return "tuple"
    if is_typeddict(inner) or is_dataclass(inner):
        return "mapping"
    return None


def has_shape(shape: str | None, value: object) -> bool:
    if shape is None:
        return True
    if shape == "int":
        return is_integer(value)
    if shape == "bool":
        return isinstance(value, bool)
    if shape == "str":
        return isinstance(value, str)
    if shape == "tuple":
        return isinstance(value, tuple)
    if shape == "mapping":
        return isinstance(value, Mapping)
    return isinstance(value, (str, Mapping))  # "field"


def any_of(branches: Sequence[tuple[object, TypeCheck]], description: str) -> TypeCheck:
    """A member whose type is a union of shapes, judged by the branch it fits.

    The branch whose top-level shape the value has is the one that
    reports -- so an element inside a malformed array is located inside
    the array, rather than the whole array being called wrong. A value
    fitting no branch's shape is reported once, by what was expected.
    """

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        fitting = [
            check for annotation, check in branches if has_shape(shape_of(annotation), value)
        ]
        if len(fitting) == 0:
            return problem(loc, f"expected {description}, got {value!r}")
        verdicts = [check(value, loc) for check in fitting]
        return () if any(len(verdict) == 0 for verdict in verdicts) else verdicts[0]

    return check


def fixed_tuple(elements: Sequence[TypeCheck], description: str) -> TypeCheck:
    """A member whose type is an array of a fixed length, checked position by position."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, tuple) or len(cast("tuple[object, ...]", value)) != len(elements):
            return problem(loc, f"expected {description}, got {value!r}")
        entries = cast("tuple[object, ...]", value)
        return tuple(
            found
            for position, (element, entry) in enumerate(zip(elements, entries, strict=True))
            for found in element(entry, (*loc, position))
        )

    return check


def mapping_of(members: Mapping[str, tuple[bool, TypeCheck]]) -> TypeCheck:
    """A member that is itself an object with declared keys, checked key by key.

    Closed, like every configuration in this package: a key the type does
    not declare is `unknown_key`, a required one missing is `missing_key`,
    both located at the object. Each present member is checked at its own
    key, so a problem inside is located inside.
    """

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, Mapping):
            return problem(loc, f"expected an object, got {value!r}")
        entries = cast("Mapping[str, object]", value)
        found: list[ValidationProblem] = []
        for key in entries:
            if key not in members:
                found.extend(problem(loc, f"unexpected key {key!r}", "unknown_key"))
        for key, (required, member) in members.items():
            if key not in entries:
                if required:
                    found.extend(problem(loc, f"missing required key {key!r}", "missing_key"))
                continue
            found.extend(member(entries[key], (*loc, key)))
        return tuple(found)

    return check


def _members_of(annotations: Mapping[str, object]) -> dict[str, tuple[bool, TypeCheck]] | None:
    """A member table for a nested object's keys; None if any key's type has no check."""
    members: dict[str, tuple[bool, TypeCheck]] = {}
    for key, annotation in annotations.items():
        check = check_for(annotation)
        if check is None:
            return None
        inner, _ = strip_annotation(annotation)
        required = get_origin(annotation) is not NotRequired and not is_optional(inner)
        members[key] = (required, check)
    return members


CheckCompiler: TypeAlias = "Callable[[object], TypeCheck | None]"
"""Turns one annotation into its type check -- or None, to decline it after all."""


@dataclass(frozen=True, slots=True)
class Registration:
    """One shape `check_for` reads: how to recognise it, and what to make of it.

    `shape` and `description` are for a shape the compiler's own logic
    does not know -- a nested metadata field, say -- so that choosing a
    union branch and naming the shape in a message work for it too.
    """

    predicate: Callable[[object], bool]
    compile: CheckCompiler
    shape: str | None = None
    description: str | None = None


_CHECK_COMPILERS: Final[list[Registration]] = []
"""The shapes `check_for` reads, consulted front to back.

The built-in shapes are appended below in the order they must be tried,
and `register_check` puts a registration in front of all of them, so the
newest one wins. `_entity` registers the nested metadata field this way,
ahead of the record shape -- an entity is a dataclass too.
"""


def register_check(
    predicate: Callable[[object], bool],
    compile: CheckCompiler,
    *,
    shape: str | None = None,
    description: str | None = None,
) -> None:
    """Teach `check_for` an annotation shape it does not read.

        Hex = NewType("Hex", str)
        register_check(lambda annotation: annotation is Hex, lambda annotation: is_hex)

    `predicate` sees the annotation with `Annotated`, `NotRequired` and
    `ReadOnly` peeled; `compile` returns the check for it, calling
    `check_for` itself for any shape inside. A registration is consulted
    before every built-in one, so a package can also replace how a
    built-in shape is judged. The same door the built-ins came through,
    which is what makes the set of shapes open rather than this module's.

    `shape` names the JSON shape the annotation admits, for choosing the
    branch of a union a value fits (`"int"`, `"str"`, `"tuple"`,
    `"mapping"`, or `"field"` for a bare name or object), and
    `description` is how a message names it; both are needed only for a
    shape the compiler's own logic does not recognise.
    """
    _CHECK_COMPILERS.insert(0, Registration(predicate, compile, shape, description))


def _builtin(predicate: Callable[[object], bool]) -> Callable[[CheckCompiler], CheckCompiler]:
    """Register a built-in shape, in the order written."""

    def append(compile: CheckCompiler) -> CheckCompiler:
        _CHECK_COMPILERS.append(Registration(predicate, compile))
        return compile

    return append


@_builtin(lambda inner: inner is int)
def _compile_int(inner: object) -> TypeCheck | None:
    return is_int


@_builtin(lambda inner: inner is bool)
def _compile_bool(inner: object) -> TypeCheck | None:
    return is_bool


@_builtin(lambda inner: inner is str)
def _compile_str(inner: object) -> TypeCheck | None:
    return is_str


@_builtin(lambda inner: inner is JSONValue)
def _compile_json_value(inner: object) -> TypeCheck | None:
    return is_json_value


@_builtin(lambda inner: get_origin(inner) is Literal)
def _compile_literal(inner: object) -> TypeCheck | None:
    # Sorted, because the order `get_args` reports is not the order the
    # `Literal` was written in: two `Literal`s over the same values
    # compare and hash equal, so the first one built anywhere in the
    # process is the one every later one resolves to. The check is a
    # membership test either way; this is so the message listing the
    # values does not depend on import order.
    return one_of(tuple(sorted(cast("tuple[str, ...]", get_args(inner)))))


@_builtin(is_union)
def _compile_union(inner: object) -> TypeCheck | None:
    branches = [arg for arg in get_args(inner) if arg is not UNSET]
    if len(branches) == 1:
        return check_for(branches[0])
    compiled = [(branch, check_for(branch)) for branch in branches]
    if any(check is None for _, check in compiled):
        return None
    return any_of(
        [(branch, cast("TypeCheck", check)) for branch, check in compiled], describe(inner)
    )


@_builtin(lambda inner: get_origin(inner) is tuple)
def _compile_tuple(inner: object) -> TypeCheck | None:
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        element = check_for(arguments[0])
        return None if element is None else sequence_of(element)
    elements = [check_for(argument) for argument in arguments]
    if any(element is None for element in elements):
        return None
    return fixed_tuple([cast("TypeCheck", element) for element in elements], describe(inner))


@_builtin(is_typeddict)
def _compile_typeddict(inner: object) -> TypeCheck | None:
    members = _members_of(get_type_hints(inner, include_extras=True))
    return None if members is None else mapping_of(members)


@_builtin(lambda inner: isinstance(inner, type) and is_dataclass(inner))
def _compile_record(inner: object) -> TypeCheck | None:
    if not isinstance(inner, type):  # pragma: no cover - the predicate says it is
        return None
    members = _members_of(field_hints(inner))
    return None if members is None else mapping_of(members)


def check_for(annotation: object) -> TypeCheck | None:
    """The type check a field annotation implies, or None if it implies none.

    A small compiler over the shapes this package's metadata takes: the
    JSON scalars, a `Literal` of names, arrays homogeneous or fixed,
    unions of those, a nested object described by a TypedDict or a
    record dataclass, and a nested metadata field -- an entity type,
    with or without `Opaque`. `UNSET` in a union says the member may be
    absent, which is the other half of a table entry and is read
    separately by `is_optional`.

    Open: each shape is a registration in `_CHECK_COMPILERS`, and
    `register_check` adds one from outside. None for an annotation no
    registration claims, which the entity then declares a check for by
    hand.
    """
    inner, _ = strip_annotation(annotation)
    for registration in _CHECK_COMPILERS:
        if registration.predicate(inner):
            return registration.compile(inner)
    return None


def derive_member_types(cls: type) -> tuple[dict[str, tuple[bool, TypeCheck]], list[str]]:
    """The member table an entity's own fields describe.

    Every field is a configuration member unless `FROM_NAME` says it is
    carried by the envelope. Requiredness is whether the type admits
    `UNSET`; the check is whatever `check_for` reads off the type. Also
    returned: the fields no check could be read for, which the entity
    must declare by hand.
    """
    derived: dict[str, tuple[bool, TypeCheck]] = {}
    unread: list[str] = []
    for name, annotation in field_hints(cls).items():
        inner, metadata = strip_annotation(annotation)
        if any(entry is FROM_NAME for entry in metadata):
            continue
        check = check_for(inner)
        if check is None:
            unread.append(name)
            continue
        derived[name] = (not is_optional(inner), check)
    return derived, unread


MemberRule: TypeAlias = "Callable[..., tuple[ValidationProblem, ...]]"
"""A rule about one member: takes its value, reports relative to it."""


_RULE_MEMBERS: Final[dict[object, tuple[str, ...]]] = {}
"""Which members each `@validates` rule is about, keyed by the function.

A side table rather than an attribute on the function, so the decorator
hands back exactly what it was given -- the declared signature survives,
and the type checker keeps checking the body and its callers.
"""


_Rule = TypeVar("_Rule", bound="Callable[..., tuple[ValidationProblem, ...]]")


def rule_members(function: object) -> tuple[str, ...]:
    """The members a function was marked as a rule about, if any."""
    return _RULE_MEMBERS.get(function, ())


def validates(*members: str) -> Callable[[_Rule], _Rule]:
    """Mark a static rule as being about one member, or several alike.

        @staticmethod
        @validates("order")
        def _order_permutes_itself(order: tuple[int, ...]) -> tuple[ValidationProblem, ...]:
            ...

    The rule receives the member's value, already of the type the field
    declares, and only when the member is present; it reports relative
    to the member, so a problem with an empty location is about the
    member itself. Naming several members applies the one rule to each.
    A rule that reads two members together is `value_problems`.
    """

    def mark(rule: _Rule) -> _Rule:
        _RULE_MEMBERS[rule] = members
        return rule

    return mark


def _bound_check(metadata: Sequence[object]) -> TypeCheck | None:
    """The check the bound markers among an annotation's metadata imply, or None."""
    ge = gt = le = lt = None
    for marker in metadata:
        if isinstance(marker, Ge):
            ge = marker.bound
        elif isinstance(marker, Gt):
            gt = marker.bound
        elif isinstance(marker, Le):
            le = marker.bound
        elif isinstance(marker, Lt):
            lt = marker.bound
        elif isinstance(marker, Interval):
            ge = marker.ge if marker.ge is not None else ge
            gt = marker.gt if marker.gt is not None else gt
            le = marker.le if marker.le is not None else le
            lt = marker.lt if marker.lt is not None else lt
    if ge is None and gt is None and le is None and lt is None:
        return None
    if ge is not None and le is not None and gt is None and lt is None:
        expectation = f"an integer in [{ge}, {le}]"
    else:
        comparisons = [
            text
            for bound, text in (
                (ge, f">= {ge}"),
                (gt, f"> {gt}"),
                (le, f"<= {le}"),
                (lt, f"< {lt}"),
            )
            if bound is not None
        ]
        expectation = "an integer " + " and ".join(comparisons)

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        # Not a number: the type check's finding, not this one's.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ()
        within_bounds = (
            (ge is None or value >= ge)
            and (gt is None or value > gt)
            and (le is None or value <= le)
            and (lt is None or value < lt)
        )
        if within_bounds:
            return ()
        return problem(loc, f"expected {expectation}, got {value}", "invalid_value")

    return check


def value_check_for(annotation: object) -> TypeCheck | None:
    """The value check an annotation's metadata implies, at any depth, or None.

    Over the shapes as the entity holds them, not as the JSON spells
    them: this runs after every member has its type and every nested
    entity has been read, so a record is a dataclass instance here and
    an entity is skipped -- it is valid by construction.
    """
    inner, metadata = strip_annotation(annotation)
    own = _bound_check(metadata)
    below: TypeCheck | None = None
    if is_union(inner):
        branches = [
            (branch, value_check_for(branch)) for branch in get_args(inner) if branch is not UNSET
        ]
        if any(check is not None for _, check in branches):

            def by_branch(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                for branch, check in branches:
                    if check is not None and has_shape(shape_of(branch), value):
                        return check(value, loc)
                return ()

            below = by_branch
    elif get_origin(inner) is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            element = value_check_for(arguments[0])
            if element is not None:
                each = element

                def per_element(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                    entries = cast("tuple[object, ...]", value)
                    return tuple(
                        found
                        for position, entry in enumerate(entries)
                        for found in each(entry, (*loc, position))
                    )

                below = per_element
        else:
            positions = [value_check_for(argument) for argument in arguments]
            if any(check is not None for check in positions):

                def per_position(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                    entries = cast("tuple[object, ...]", value)
                    return tuple(
                        found
                        for position, (check, entry) in enumerate(
                            zip(positions, entries, strict=True)
                        )
                        if check is not None
                        for found in check(entry, (*loc, position))
                    )

                below = per_position
    elif isinstance(inner, type) and is_dataclass(inner) and shape_of(inner) != "field":
        members = {
            name: check
            for name, field_annotation in field_hints(inner).items()
            if (check := value_check_for(field_annotation)) is not None
        }
        if len(members) != 0:

            def per_field(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                return tuple(
                    found
                    for name, check in members.items()
                    if (held := getattr(value, name)) is not UNSET
                    for found in check(held, (*loc, name))
                )

            below = per_field
    if own is None and below is None:
        return None
    if below is None:
        return own
    if own is None:
        return below
    outer, inner_check = own, below

    def both(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        return (*outer(value, loc), *inner_check(value, loc))

    return both


def element_annotations(inner: object, count: int) -> list[object]:
    """The annotation of each element of a tuple type, one per element held."""
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return [arguments[0]] * count
    return list(arguments)


def is_class_var(annotation: object) -> bool:
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


def declared_class_vars(cls: type) -> dict[str, type]:
    """Every class variable annotated anywhere in `cls`'s ancestry.

    Mapped to the class that annotated it, so a message can say where the
    requirement comes from. Base first, so a redeclaration names the
    nearest ancestor.
    """
    found: dict[str, type] = {}
    for ancestor in reversed(cls.__mro__):
        for name, annotation in own_annotations(ancestor).items():
            if is_class_var(annotation):
                found[name] = ancestor
    return found


__all__ = [
    "FROM_NAME",
    "CheckCompiler",
    "Ge",
    "Gt",
    "Interval",
    "Le",
    "Lt",
    "MemberRule",
    "any_of",
    "check_for",
    "declared_class_vars",
    "derive_member_types",
    "describe",
    "element_annotations",
    "field_hints",
    "fixed_tuple",
    "has_shape",
    "is_class_var",
    "is_optional",
    "is_union",
    "mapping_of",
    "own_annotations",
    "register_check",
    "rule_members",
    "shape_of",
    "strip_annotation",
    "validates",
    "value_check_for",
]
