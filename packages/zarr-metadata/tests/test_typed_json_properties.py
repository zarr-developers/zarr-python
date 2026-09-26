"""The checker against a reference, over TypedDicts and JSON values drawn at random.

A spec is drawn first: a small description of a type in the checker's
vocabulary -- scalars, `Literal`s, arrays, unions, `Mapping[str, V]`,
`NewType`s and aliases, and TypedDicts with up to two bases, each class
with its own `total`, its keys' qualifiers written in any order, and
what it says of other keys, some holding themselves. It is written out
as source text and built from that twice: with each class's annotations
evaluated, and left as the strings `from __future__ import annotations`
leaves. Beside it stands `conforms`, a predicate over JSON worked out
from the spec alone by the typing spec's rules, not the checker's.

Values are JSON, their depth capped: drawn from the spec, so they
conform; drawn at large, so most do not; or drawn from the spec and
changed in one place, so they nearly do. On every one the checker has to
agree with the reference, and the two builds with each other.
"""

from __future__ import annotations

import copy
import itertools
import sys
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal, NewType, NotRequired, Required, TypeAlias, Union, cast

import pytest
from hypothesis import HealthCheck, event, find, given, note, settings
from hypothesis import strategies as st
from typing_extensions import ReadOnly, TypeAliasType, TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata._typed_json import Loc, Parsed, no_leaf, parser, typeddict_keys

# --- specs -----------------------------------------------------------------

Scalar: TypeAlias = Literal["int", "float", "bool", "str", "null", "json"]


@dataclass(frozen=True)
class Leaf:
    name: Scalar


@dataclass(frozen=True)
class OneOf:
    values: tuple[str | int | bool | None, ...]


@dataclass(frozen=True)
class ArrayOf:
    element: Spec


@dataclass(frozen=True)
class Fixed:
    elements: tuple[Spec, ...]


@dataclass(frozen=True)
class AnyOf:
    branches: tuple[Spec, ...]


@dataclass(frozen=True)
class MappingOf:
    value: Spec


@dataclass(frozen=True)
class Named:
    how: Literal["newtype", "alias"]
    inner: Spec


@dataclass(frozen=True)
class Itself:
    """The TypedDict a member sits in: a class that holds itself."""


@dataclass(frozen=True)
class ExtraItems:
    spec: Spec


Says: TypeAlias = Literal["unsaid", "closed", "open"] | ExtraItems
"""What a class says of the keys it does not declare: nothing, `closed=True`, `closed=False`, or `extra_items=`."""


@dataclass(frozen=True)
class Member:
    key: str
    spec: Spec
    wrappers: tuple[str, ...]
    """`Required`, `NotRequired`, `ReadOnly` and `Annotated`, as drawn, innermost first."""


@dataclass(frozen=True)
class Layer:
    """One class in a TypedDict's ancestry: its own keys and keywords, and how its module writes it."""

    total: bool
    members: tuple[Member, ...]
    says: Says
    postponed: bool = False
    """In a mixed build, whether the class's annotations are left as strings."""
    quoted: bool = False
    """Evaluated, whether the class writes the name of another class inside an annotation as a string."""


@dataclass(frozen=True)
class Object:
    layers: tuple[Layer, ...]
    """Base first; the last is the TypedDict itself. Each is written in a module of its own."""
    stdlib: bool = False
    """Whether the classes are `typing.TypedDict`s rather than `typing_extensions` ones."""


Spec: TypeAlias = Leaf | OneOf | ArrayOf | Fixed | AnyOf | MappingOf | Named | Itself | Object


def members_of(spec: Object) -> dict[str, tuple[Spec, bool]]:
    """Each key the TypedDict declares, its spec, and whether it is required, by the typing spec.

    A `Required` or `NotRequired` qualifier says; otherwise the `total` of
    the class that declared the key does.
    """
    found: dict[str, tuple[Spec, bool]] = {}
    for layer in spec.layers:
        for member in layer.members:
            found[member.key] = (member.spec, _required(member, layer.total))
    return found


def _required(member: Member, total: bool) -> bool:
    if "Required" in member.wrappers:
        return True
    if "NotRequired" in member.wrappers:
        return False
    return total


def says_of(spec: Object) -> Says:
    """What the TypedDict says of other keys: the nearest class that says anything, as the spec has it."""
    said: Says = "unsaid"
    for layer in spec.layers:
        if layer.says != "unsaid":
            said = layer.says
    return said


def conforms(value: object, spec: Spec, itself: Object | None = None) -> bool:
    """Whether JSON `value` is a value of `spec`: the reference, worked out from the spec alone."""
    if isinstance(spec, Leaf):
        return {
            "int": type(value) is int,
            "float": type(value) in (int, float),
            "bool": type(value) is bool,
            "str": type(value) is str,
            "null": value is None,
            "json": True,
        }[spec.name]
    if isinstance(spec, OneOf):
        return any(value == entry and type(value) is type(entry) for entry in spec.values)
    if isinstance(spec, ArrayOf):
        return isinstance(value, (list, tuple)) and all(
            conforms(entry, spec.element, itself) for entry in value
        )
    if isinstance(spec, Fixed):
        return (
            isinstance(value, (list, tuple))
            and len(value) == len(spec.elements)
            and all(
                conforms(entry, element, itself)
                for entry, element in zip(value, spec.elements, strict=True)
            )
        )
    if isinstance(spec, AnyOf):
        return any(conforms(value, branch, itself) for branch in spec.branches)
    if isinstance(spec, MappingOf):
        return isinstance(value, dict) and all(
            conforms(entry, spec.value, itself) for entry in value.values()
        )
    if isinstance(spec, Named):
        return conforms(value, spec.inner, itself)
    if isinstance(spec, Itself):
        assert itself is not None
        return conforms(value, itself, itself)
    if not isinstance(value, dict):
        return False
    members = members_of(spec)
    for key, (member, required) in members.items():
        if key in value:
            if not conforms(value[key], member, spec):
                return False
        elif required:
            return False
    others = [key for key in value if key not in members]
    said = says_of(spec)
    if said == "closed":
        return len(others) == 0
    if isinstance(said, ExtraItems):
        return all(conforms(value[key], said.spec, spec) for key in others)
    return True


# --- building a spec, evaluated or postponed --------------------------------

_PRELUDE: dict[str, object] = {
    "Annotated": Annotated,
    "JSONValue": JSONValue,
    "Literal": Literal,
    "Mapping": Mapping,
    "NewType": NewType,
    "NotRequired": NotRequired,
    "ReadOnly": ReadOnly,
    "Required": Required,
    "TypeAliasType": TypeAliasType,
    "Union": Union,
}
_serial = itertools.count()

_LEAF_TEXT: dict[str, str] = {
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
    "null": "None",
    "json": "JSONValue",
}

Mode: TypeAlias = Literal["evaluated", "postponed", "mixed"]
"""How a build writes its classes' annotations: all evaluated, all strings, or as each class says."""

_ONE_MODULE_FOR_STDLIB = sys.version_info < (3, 12)
"""`typing.TypedDict` records a class's bases from 3.12; before, a subclass reads its keys in one module."""


def _module() -> types.ModuleType:
    """A fresh module, registered where a string annotation is resolved, knowing only the prelude."""
    module = types.ModuleType(f"tests._typed_json_properties_{next(_serial)}")
    module.__dict__.update(_PRELUDE)
    sys.modules[module.__name__] = module
    return module


def _evaluated(text: str, home: types.ModuleType) -> object:
    return eval(text, home.__dict__)


@dataclass(frozen=True)
class Build:
    """A spec built as source, each class in a module of its own, the names it uses imported there."""

    mode: Mode
    source: list[str] = field(default_factory=list)
    """What was built, as it would be written: for a failing example's notes."""

    def annotation(self, spec: Spec) -> object:
        home = _module()
        return _evaluated(self.text(spec, None, home, quoted=False), home)

    def _named(self, name: str, made: object, home: types.ModuleType, *, quoted: bool) -> str:
        home.__dict__[name] = made
        return repr(name) if quoted else name

    def text(self, spec: Spec, itself: str | None, home: types.ModuleType, *, quoted: bool) -> str:
        """`spec` written in `home`, a name inside another annotation quoted when `quoted`."""
        if isinstance(spec, Leaf):
            return _LEAF_TEXT[spec.name]
        if isinstance(spec, OneOf):
            return f"Literal[{', '.join(repr(value) for value in spec.values)}]"
        if isinstance(spec, ArrayOf):
            return f"tuple[{self.text(spec.element, itself, home, quoted=quoted)}, ...]"
        if isinstance(spec, Fixed):
            elements = ", ".join(
                self.text(element, itself, home, quoted=quoted) for element in spec.elements
            )
            return f"tuple[{elements}]"
        if isinstance(spec, AnyOf):
            branches = ", ".join(
                self.text(branch, itself, home, quoted=quoted) for branch in spec.branches
            )
            return f"Union[{branches}]"
        if isinstance(spec, MappingOf):
            return f"Mapping[str, {self.text(spec.value, itself, home, quoted=quoted)}]"
        if isinstance(spec, Named):
            name = f"Named{next(_serial)}"
            maker = "NewType" if spec.how == "newtype" else "TypeAliasType"
            inner = self.text(spec.inner, itself, home, quoted=quoted and spec.how == "alias")
            written = f"{maker}({name!r}, {inner})"
            self.source.append(f"# {home.__name__}\n{name} = {written}")
            return self._named(name, _evaluated(written, home), home, quoted=quoted)
        if isinstance(spec, Itself):
            assert itself is not None
            return repr(itself)  # a forward reference: the class is not built yet
        name, built = self.typeddict(spec)
        return self._named(name, built, home, quoted=quoted)

    def member(self, member: Member, itself: str, home: types.ModuleType, *, quoted: bool) -> str:
        text = self.text(member.spec, itself, home, quoted=quoted)
        for wrapper in member.wrappers:
            text = f"Annotated[{text}, 'meta']" if wrapper == "Annotated" else f"{wrapper}[{text}]"
        return text

    def typeddict(self, spec: Object) -> tuple[str, object]:
        """The TypedDict `spec` is, a class per layer, each in its module: its name, and it."""
        names = [f"T{next(_serial)}" for _ in spec.layers]
        shared = _module() if spec.stdlib and _ONE_MODULE_FOR_STDLIB else None
        base: object = typing.TypedDict if spec.stdlib else TypedDict
        base_name = "typing.TypedDict" if spec.stdlib else "TypedDict"
        for layer, name in zip(spec.layers, names, strict=True):
            home = shared or _module()
            postponed = self.mode == "postponed" or (self.mode == "mixed" and layer.postponed)
            quoted = not postponed and layer.quoted
            texts = {
                member.key: self.member(member, names[-1], home, quoted=quoted)
                for member in layer.members
            }
            annotations = (
                texts if postponed else {key: _evaluated(text, home) for key, text in texts.items()}
            )
            keywords: dict[str, object] = {"total": layer.total}
            if layer.says == "closed":
                keywords["closed"] = True
            elif layer.says == "open":
                keywords["closed"] = False
            elif isinstance(layer.says, ExtraItems):
                extra = self.text(layer.says.spec, None, home, quoted=False)
                keywords["extra_items"] = extra if postponed else _evaluated(extra, home)
            namespace = {"__annotations__": annotations, "__module__": home.__name__}
            built = types.new_class(name, (base,), keywords, lambda body: body.update(namespace))  # noqa: B023 - run at once
            home.__dict__[name] = built
            written = "".join(
                f"\n    {key}: {text!r}" if postponed else f"\n    {key}: {text}"
                for key, text in texts.items()
            )
            header = "from __future__ import annotations\n" if postponed else ""
            self.source.append(
                f"# {home.__name__}\n{header}class {name}({base_name}, {keywords!r}):{written or ' ...'}"
            )
            base, base_name = built, name
        return names[-1], base

    def text_of(self) -> str:
        return "\n\n".join(self.source)


# --- strategies ------------------------------------------------------------

_KEYS = ("a", "b", "c", "d", "x-y")
_LITERALS: tuple[str | int | bool | None, ...] = (None, True, False, 0, 1, -1, "", "a", "b")
_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-2, 2),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=2),
)
_leaves = st.one_of(
    st.builds(Leaf, st.sampled_from(("int", "float", "bool", "str", "null", "json"))),
    st.lists(
        st.sampled_from(_LITERALS), min_size=1, max_size=3, unique_by=lambda v: (type(v), v)
    ).map(lambda values: OneOf(tuple(values))),
)


@st.composite
def _wrappers(draw: st.DrawFn) -> tuple[str, ...]:
    chosen = [draw(st.sampled_from(("", "Required", "NotRequired")))]
    chosen += ["ReadOnly"] if draw(st.booleans()) else []
    chosen += ["Annotated"] if draw(st.booleans()) else []
    return tuple(draw(st.permutations([wrapper for wrapper in chosen if wrapper != ""])))


_TYPING_TAKES_PEP_728 = sys.version_info >= (3, 15)
"""Whether `typing.TypedDict` takes `closed=` and `extra_items=`, as `typing_extensions` does."""


@st.composite
def _objects(draw: st.DrawFn, inner: st.SearchStrategy[Spec]) -> Object:
    keys = draw(st.lists(st.sampled_from(_KEYS), unique=True, max_size=4))
    count = draw(st.integers(1, 3))
    owners = [draw(st.integers(0, count - 1)) for _ in keys]
    layers: list[Layer] = []
    said: Says = "unsaid"
    declared: dict[str, tuple[Member, bool]] = {}
    for index in range(count):
        # A subclass of a closed TypedDict, or of one with extra items, is
        # kept to what the typing spec allows it: to say the same, or
        # nothing, and to add no key.
        restricted = said == "closed" or isinstance(said, ExtraItems)
        choices: list[Says] = ["unsaid", said] if restricted else ["unsaid", "closed", "open"]
        if not restricted and draw(st.booleans()):
            choices.append(ExtraItems(draw(inner)))
        says = draw(st.sampled_from(choices))
        total = draw(st.booleans())
        members = [
            Member(key, draw(inner), draw(_wrappers()))
            for key, owner in zip(keys, owners, strict=True)
            if owner == index and not restricted
        ]
        if not restricted:
            members += draw(_redeclared(declared, total))
        if index == count - 1 and not restricted and draw(st.booleans()):
            members.append(Member("self", ArrayOf(Itself()), draw(_wrappers())))
        postponed, quoted = draw(st.booleans()), draw(st.booleans())
        layers.append(Layer(total, tuple(members), says, postponed, quoted))
        declared.update({member.key: (member, _required(member, total)) for member in members})
        said = says if says != "unsaid" else said
    speaks = any(layer.says != "unsaid" for layer in layers)
    # `typing.TypedDict` from 3.13 refuses a read-only key redeclaring one it
    # thinks mutable, and cannot see `ReadOnly` in a postponed base: it would
    # not build what the spec allows, so it declares no key twice here.
    redeclares = sum(len(layer.members) for layer in layers) > len(
        {member.key for layer in layers for member in layer.members}
    )
    stdlib = (_TYPING_TAKES_PEP_728 or not speaks) and not redeclares and draw(st.booleans())
    return Object(tuple(layers), stdlib)


@st.composite
def _redeclared(
    draw: st.DrawFn, declared: dict[str, tuple[Member, bool]], total: bool
) -> list[Member]:
    """None, or one read-only key a base declared, narrowed as the typing spec lets a subclass.

    Its type may narrow -- a union to some of its branches -- and a key
    that was not required may become required, never the other way.
    """
    candidates = sorted(
        key for key, (member, _) in declared.items() if "ReadOnly" in member.wrappers
    )
    if len(candidates) == 0 or not draw(st.booleans()):
        return []
    key = draw(st.sampled_from(candidates))
    earlier, was_required = declared[key]
    spec = earlier.spec
    if isinstance(spec, AnyOf):
        kept = sorted(draw(st.sets(st.integers(0, len(spec.branches) - 1), min_size=1)))
        branches = tuple(spec.branches[index] for index in kept)
        spec = branches[0] if len(branches) == 1 else AnyOf(branches)
    wrappers = draw(_wrappers())
    if was_required and not _required(Member(key, spec, wrappers), total):
        wrappers = ("Required", *(wrapper for wrapper in wrappers if wrapper != "NotRequired"))
    return [Member(key, spec, wrappers)]


def specs(depth: int) -> st.SearchStrategy[Spec]:
    if depth == 0:
        return _leaves
    inner = specs(depth - 1)
    return st.one_of(
        _leaves,
        st.builds(ArrayOf, inner),
        st.lists(inner, min_size=1, max_size=3).map(lambda elements: Fixed(tuple(elements))),
        st.lists(inner, min_size=2, max_size=3).map(lambda branches: AnyOf(tuple(branches))),
        st.builds(MappingOf, inner),
        st.builds(Named, st.sampled_from(("newtype", "alias")), inner),
        _objects(inner),
    )


def json_values(depth: int) -> st.SearchStrategy[object]:
    """Any JSON, nested no deeper than `depth`."""
    if depth <= 0:
        return _SCALARS
    inner = json_values(depth - 1)
    keys = st.sampled_from((*_KEYS, "self", "z")) | st.text(max_size=2)
    return st.one_of(
        _SCALARS, st.lists(inner, max_size=2), st.dictionaries(keys, inner, max_size=2)
    )


def conforming(spec: Spec, depth: int, itself: Object | None = None) -> st.SearchStrategy[object]:
    """JSON values of `spec`, arrays and objects of any key nested no deeper than `depth`."""
    if isinstance(spec, Leaf):
        return {
            "int": st.integers(),
            "float": st.integers() | st.floats(allow_nan=False, allow_infinity=False),
            "bool": st.booleans(),
            "str": st.text(max_size=3),
            "null": st.none(),
            "json": json_values(depth),
        }[spec.name]
    if isinstance(spec, OneOf):
        return st.sampled_from(spec.values)
    if isinstance(spec, ArrayOf):
        if depth <= 0:
            return st.just([])
        return st.lists(conforming(spec.element, depth - 1, itself), max_size=2)
    if isinstance(spec, Fixed):
        elements = [conforming(element, depth - 1, itself) for element in spec.elements]
        return st.tuples(*elements).map(list)
    if isinstance(spec, AnyOf):
        return st.one_of(*(conforming(branch, depth, itself) for branch in spec.branches))
    if isinstance(spec, MappingOf):
        if depth <= 0:
            return st.just({})
        return st.dictionaries(
            st.text(max_size=2), conforming(spec.value, depth - 1, itself), max_size=2
        )
    if isinstance(spec, Named):
        return conforming(spec.inner, depth, itself)
    if isinstance(spec, Itself):
        assert itself is not None
        return conforming(itself, depth, itself)
    return _object_values(spec, depth)


@st.composite
def _object_values(draw: st.DrawFn, spec: Object, depth: int) -> dict[str, object]:
    members = members_of(spec)
    value: dict[str, object] = {}
    for key, (member, required) in members.items():
        if required or draw(st.booleans()):
            value[key] = draw(conforming(member, depth - 1, spec))
    said = says_of(spec)
    others = st.text(max_size=3).filter(lambda key: key not in members)
    if said in ("unsaid", "open"):
        value.update(draw(st.dictionaries(others, json_values(depth - 1), max_size=2)))
    elif isinstance(said, ExtraItems):
        value.update(
            draw(st.dictionaries(others, conforming(said.spec, depth - 1, spec), max_size=2))
        )
    return value


def _positions(value: object, path: Loc = ()) -> list[Loc]:
    """Every place in `value`: itself, and each position inside an array or object, depth first."""
    found: list[Loc] = [path]
    if isinstance(value, dict):
        for key, entry in value.items():
            found += _positions(entry, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, entry in enumerate(value):
            found += _positions(entry, (*path, index))
    return found


def _changed(value: object, path: Loc, change: object) -> object:
    """`value` with what sits at `path` replaced by `change`, or removed if `change` is `_GONE`."""
    if len(path) == 0:
        return change
    step, rest = path[0], path[1:]
    if isinstance(value, dict):
        entries = dict(value)
        if len(rest) == 0 and change is _GONE:
            del entries[step]
        else:
            entries[step] = _changed(entries[step], rest, change)
        return entries
    assert isinstance(value, (list, tuple))
    assert isinstance(step, int)
    entries_ = list(value)
    if len(rest) == 0 and change is _GONE:
        del entries_[step]
    else:
        entries_[step] = _changed(entries_[step], rest, change)
    return entries_


_GONE = object()


@st.composite
def nearly(draw: st.DrawFn, spec: Spec, depth: int) -> object:
    """A value of `spec`, changed in one place: a value replaced, a key or entry dropped, or one added."""
    value = draw(conforming(spec, depth))
    path = draw(st.sampled_from(_positions(value)))
    how = draw(st.sampled_from(("replace", "drop", "add")))
    if how == "drop" and len(path) != 0:
        return _changed(value, path, _GONE)
    target = _at(value, path)
    if how == "add" and isinstance(target, dict):
        key = draw(st.sampled_from((*_KEYS, "self", "z")))
        return _changed(value, path, {**target, key: draw(json_values(1))})
    if how == "add" and isinstance(target, list):
        return _changed(value, path, [*target, draw(json_values(1))])
    return _changed(value, path, draw(json_values(1)))


def _at(value: object, loc: Loc) -> object:
    for step in loc:
        if isinstance(value, dict):
            value = cast("dict[str, object]", value)[cast("str", step)]
        else:
            value = cast("list[object]", value)[cast("int", step)]
    return value


def _is_place(value: object, loc: Loc) -> bool:
    """Whether `loc` names a place in `value`."""
    for step in loc:
        in_object = isinstance(value, dict) and isinstance(step, str) and step in value
        in_array = (
            isinstance(value, (list, tuple)) and isinstance(step, int) and 0 <= step < len(value)
        )
        if not (in_object or in_array):
            return False
        value = _at(value, (step,))
    return True


def _same_json(left: object, right: object) -> bool:
    """Equal as JSON: an array is an array, list or tuple."""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(map(_same_json, left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json(left[key], right[key]) for key in left
        )
    return type(left) is type(right) and left == right


# --- the properties ---------------------------------------------------------

_MODES: tuple[Mode, ...] = ("evaluated", "postponed", "mixed")

_DEPTH = 3
"""The operational cap: how deep a spec nests, and how deep a value of it."""

_EXAMPLES = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _values(spec: Spec) -> st.SearchStrategy[object]:
    return st.one_of(conforming(spec, _DEPTH), nearly(spec, _DEPTH), json_values(_DEPTH))


def _read(annotation: object, value: object) -> Parsed:
    return parser(annotation, no_leaf)(value, ())


@_EXAMPLES
@given(st.data())
def test_the_checker_agrees_with_the_reference(data: st.DataObject) -> None:
    # No problem exactly when the value is one of the type; the value that
    # comes back is the one read, as JSON; every problem is located at a
    # place in the value, or, for a missing key, in the object that lacks
    # it; and when every problem is a key a closed TypedDict does not
    # declare, what comes back is a value of the type.
    spec = data.draw(specs(_DEPTH), label="spec")
    build = Build(data.draw(st.sampled_from(_MODES), label="mode"))
    annotation = build.annotation(spec)
    note(build.text_of())
    value = data.draw(_values(spec), label="value")
    event(f"spec {type(spec).__name__}")
    before = copy.deepcopy(value)
    typed, problems = _read(annotation, value)
    expected = conforms(value, spec)
    event("conforms" if expected else "does not conform")
    assert value == before
    assert (len(problems) == 0) == expected, problems
    if expected:
        assert _same_json(typed, value)
    for found in problems:
        place = found.loc[:-1] if found.kind == "missing_key" else found.loc
        assert _is_place(value, place), found
    if len(problems) != 0 and all(found.kind == "unknown_key" for found in problems):
        event("only unknown keys")
        assert conforms(typed, spec)


@_EXAMPLES
@given(st.data())
def test_a_postponed_typeddict_reads_as_an_evaluated_one(data: st.DataObject) -> None:
    # The same classes, their annotations evaluated or left as strings, as
    # `from __future__ import annotations` leaves them: read alike, down to
    # each problem's message.
    spec = data.draw(specs(_DEPTH), label="spec")
    evaluated = Build("evaluated")
    written = Build(data.draw(st.sampled_from(("postponed", "mixed")), label="mode"))
    read_evaluated, read_written = evaluated.annotation(spec), written.annotation(spec)
    note(evaluated.text_of())
    note(written.text_of())
    value = data.draw(_values(spec), label="value")
    assert _read(read_evaluated, value) == _read(read_written, value)


@_EXAMPLES
@given(st.data())
def test_typeddict_keys_follow_the_typing_spec(data: st.DataObject) -> None:
    # Which keys are required, per class `total` under each key's own
    # qualifiers, and what other keys may hold, inherited when a class
    # says nothing: as the spec reads the declaration, evaluated or not.
    spec = data.draw(_objects(specs(_DEPTH - 1)), label="spec")
    members = members_of(spec)
    said = says_of(spec)
    event(f"{len(spec.layers)} classes")
    event(f"says {said if isinstance(said, str) else 'extra_items'}")
    for mode in _MODES:
        build = Build(mode)
        keys = typeddict_keys(cast("type", build.annotation(spec)))
        note(build.text_of())
        assert set(keys.members) == set(members)
        assert keys.required == {key for key, (_, required) in members.items() if required}
        assert keys.closed == (said == "closed")
        assert keys.open == (said in ("unsaid", "open"))
        assert keys.declared == (said != "unsaid")


@_EXAMPLES
@given(st.data())
def test_a_key_a_closed_typeddict_does_not_declare_is_reported_and_left_out(
    data: st.DataObject,
) -> None:
    # The value still reads, as the value it was before the key was added;
    # the key is the one problem, where it sits.
    spec = data.draw(
        _objects(specs(_DEPTH - 1)).filter(lambda drawn: says_of(drawn) == "closed"), label="spec"
    )
    declared = members_of(spec)
    value = data.draw(_object_values(spec, _DEPTH), label="value")
    key = data.draw(st.text(max_size=3).filter(lambda drawn: drawn not in declared), label="key")
    build = Build(data.draw(st.sampled_from(_MODES), label="mode"))
    annotation = build.annotation(spec)
    note(build.text_of())
    typed, problems = _read(annotation, {**value, key: data.draw(json_values(1), label="held")})
    assert [(found.loc, found.kind) for found in problems] == [((key,), "unknown_key")]
    assert _same_json(typed, value)


# --- reach: what the strategy has to keep drawing -----------------------------


def _nests_a_name(spec: Spec, *, nested: bool = False) -> bool:
    """Whether `spec` names another class or alias inside an annotation, where a quoted name loses its module."""
    if isinstance(spec, (Named, Object)):
        return nested
    if isinstance(spec, ArrayOf):
        return _nests_a_name(spec.element, nested=True)
    if isinstance(spec, MappingOf):
        return _nests_a_name(spec.value, nested=True)
    if isinstance(spec, Fixed):
        return any(_nests_a_name(element, nested=True) for element in spec.elements)
    if isinstance(spec, AnyOf):
        return any(_nests_a_name(branch, nested=True) for branch in spec.branches)
    return False


def _hides_a_qualifier(spec: Object) -> bool:
    """Whether a postponed class writes a qualifier its `total` contradicts, which the runtime then misses."""
    return any(
        layer.postponed
        and any(
            ("NotRequired" in member.wrappers and layer.total)
            or ("Required" in member.wrappers and not layer.total)
            for member in layer.members
        )
        for layer in spec.layers
    )


_REACHES = {
    "a-key-redeclared": lambda spec: any(
        member.key in {earlier.key for layer in spec.layers[:index] for earlier in layer.members}
        for index, later in enumerate(spec.layers)
        for member in later.members
    ),
    "openness-inherited": lambda spec: (
        spec.layers[-1].says == "unsaid" and any(layer.says != "unsaid" for layer in spec.layers)
    ),
    "typing-typeddict": lambda spec: spec.stdlib and len(spec.layers) > 1,
    "classes-evaluated-and-postponed": lambda spec: (
        len({layer.postponed for layer in spec.layers}) == 2
    ),
    "a-qualifier-the-runtime-cannot-see": _hides_a_qualifier,
    "a-quoted-name-a-subclass-inherits": lambda spec: (
        len(spec.layers) > 1
        and any(
            layer.quoted and any(_nests_a_name(member.spec) for member in layer.members)
            for layer in spec.layers[:-1]
        )
    ),
    "a-class-that-holds-itself": lambda spec: any(
        member.key == "self" for layer in spec.layers for member in layer.members
    ),
}


@pytest.mark.parametrize("reach", sorted(_REACHES))
def test_the_strategy_reaches(reach: str) -> None:
    # Each case the properties are meant to cover, drawn at least once in a
    # modest search: an edit to the strategy cannot quietly lose one.
    find(
        _objects(specs(_DEPTH - 1)),
        _REACHES[reach],
        settings=settings(max_examples=2000, database=None, deadline=None),
    )
