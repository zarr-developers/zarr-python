"""The checker: parsers compiled from annotations for the shapes JSON takes.

Tested on its own, with no metadata field in sight. A leaf stands in for
whatever a caller adds -- the definition layer adds the shape of a member
holding another metadata field -- and `no_leaf` adds nothing.
"""

from __future__ import annotations

import itertools
import sys
import textwrap
import types
from collections.abc import Mapping
from typing import Generic, Literal, Never, NewType, NotRequired, TypeVar

import pytest
from typing_extensions import ReadOnly, TypeAliasType, TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._typed_json import (
    Loc,
    Parsed,
    Parser,
    describe,
    no_leaf,
    parser,
    parser_for,
    shape_of,
    typeddict_keys,
)

Level = NewType("Level", int)
Width = TypeAliasType("Width", int)


class Options(TypedDict, closed=True):
    level: int
    note: NotRequired[str]
    tag: ReadOnly[NotRequired[str]]


class WithExtras(TypedDict, extra_items=int):
    level: int


class Open(TypedDict, closed=False):
    level: int


class Small(TypedDict, closed=True):
    x: int


class Large(TypedDict, closed=True):
    x: int
    y: int


class Tree(TypedDict, closed=True):
    label: str
    children: tuple[Tree, ...]


def _read(annotation: object, value: object) -> Parsed:
    return parser(annotation, no_leaf)(value, ())


def _found(parsed: Parsed) -> list[tuple[Loc, str]]:
    return [(found.loc, found.kind) for found in parsed[1]]


@pytest.mark.parametrize(
    ("annotation", "value", "parsed"),
    [
        (int, 3, 3),
        (float, 3, 3),
        (float, 2.5, 2.5),
        (bool, True, True),
        (str, "x", "x"),
        (None, None, None),
        (Literal["a", "b"], "a", "a"),
        (Literal[1, 2], 2, 2),
        (tuple[int, ...], [1, 2], (1, 2)),
        (tuple[int, str], (1, "a"), (1, "a")),
        (tuple[str | None, ...], ["a", None], ("a", None)),
        (int | str, "a", "a"),
        (Mapping[str, int], {"k": 1}, {"k": 1}),
        (JSONValue, {"any": [1, None]}, {"any": [1, None]}),
        (Level, 5, 5),
        (Width, 5, 5),
    ],
    ids=[
        "int",
        "number-from-int",
        "number",
        "bool",
        "str",
        "null",
        "literal-str",
        "literal-int",
        "sequence",
        "fixed-tuple",
        "sequence-of-optional",
        "union",
        "mapping",
        "json",
        "newtype",
        "alias",
    ],
)
def test_a_value_of_the_shape_reads_as_itself(
    annotation: object, value: object, parsed: object
) -> None:
    # The shapes JSON takes and no others, each compiled once from its
    # annotation; an array comes back as a tuple, the rest as it came.
    assert _read(annotation, value) == (parsed, ())


@pytest.mark.parametrize(
    ("annotation", "value", "loc", "kind"),
    [
        (int, True, (), "invalid_type"),
        (int, 2.0, (), "invalid_type"),
        (float, "2", (), "invalid_type"),
        (str, 1, (), "invalid_type"),
        (None, 0, (), "invalid_type"),
        (Literal["a"], "b", (), "invalid_value"),
        (Literal[1], True, (), "invalid_value"),
        (tuple[int, ...], (1, "x"), (1,), "invalid_type"),
        (tuple[int, ...], 5, (), "invalid_type"),
        (int | str, 2.5, (), "invalid_type"),
        (Mapping[str, int], {"k": "v"}, ("k",), "invalid_type"),
        (JSONValue, {"k": object()}, (), "invalid_type"),
    ],
    ids=[
        "bool-is-not-int",
        "float-is-not-int",
        "str-is-not-number",
        "int-is-not-str",
        "zero-is-not-null",
        "literal-other",
        "literal-bool-is-not-int",
        "element",
        "not-a-sequence",
        "no-branch",
        "mapping-value",
        "not-json",
    ],
)
def test_error_a_value_of_another_shape_is_located(
    annotation: object, value: object, loc: Loc, kind: str
) -> None:
    parsed, problems = _read(annotation, value)
    assert parsed == value
    assert [(problem.loc, problem.kind) for problem in problems] == [(loc, kind)]


@pytest.mark.parametrize(
    ("annotation", "value", "parsed", "found"),
    [
        (Options, {"level": 1}, {"level": 1}, []),
        (Options, {"note": "n", "level": 1}, {"note": "n", "level": 1}, []),
        (Options, {"level": 1, "extra": 2}, {"level": 1}, [(("extra",), "unknown_key")]),
        (WithExtras, {"level": 1, "more": 2}, {"level": 1, "more": 2}, []),
        (
            WithExtras,
            {"level": 1, "more": "two"},
            {"level": 1, "more": "two"},
            [(("more",), "invalid_type")],
        ),
        (Open, {"level": 1, "more": [2]}, {"level": 1, "more": [2]}, []),
        (Small | Large, {"x": 1, "y": 2}, {"x": 1, "y": 2}, []),
        (Small | Large, {"x": 1, "z": 2}, {"x": 1}, [(("z",), "unknown_key")]),
        (
            Tree,
            {"label": "a", "children": [{"label": "b", "children": []}]},
            {"label": "a", "children": ({"label": "b", "children": ()},)},
            [],
        ),
        (
            Tree,
            {"label": "a", "children": [{"label": 1, "children": []}]},
            {"label": "a", "children": ({"label": 1, "children": ()},)},
            [(("children", 0, "label"), "invalid_type")],
        ),
    ],
    ids=[
        "closed",
        "closed-keeps-the-order-keys-came-in",
        "closed-leaves-an-unknown-key-out",
        "extra-items",
        "extra-items-of-the-wrong-type",
        "open",
        "union-takes-the-branch-that-accepts",
        "union-falls-back-to-the-first-that-tried",
        "recursive",
        "recursive-located",
    ],
)
def test_an_object_reads_as_what_its_typeddict_admits(
    annotation: object, value: object, parsed: dict[str, object], found: list[tuple[Loc, str]]
) -> None:
    # A new dict of the keys the type admits, in the order they came: a
    # closed TypedDict reports a key it does not declare and leaves it
    # out, `extra_items` types every other key, an open one keeps them.
    result = _read(annotation, value)
    assert result[0] == parsed
    assert list(cast_dict(result[0])) == list(parsed)
    assert _found(result) == found


def cast_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def test_error_a_typed_dict_misses_a_required_key() -> None:
    value, problems = parser(Options, no_leaf)({"note": "n"}, ())
    assert value == {"note": "n"}
    assert [(problem.loc, problem.kind) for problem in problems] == [(("level",), "missing_key")]


# --- a TypedDict, read as the typing spec defines it -----------------------

_PRELUDE = """\
import typing
from typing import Annotated, Generic, NotRequired, Required, TypeVar
from typing_extensions import Never, ReadOnly, TypedDict
V = TypeVar("V")
"""

_modules = itertools.count()


def _declared(source: str, *, postponed: bool, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The module `source` defines, written with or without `from __future__ import annotations`.

    Registered in `sys.modules`, because a string annotation is resolved
    in the module its class was defined in.
    """
    name = f"tests.v3._typeddicts_{next(_modules)}"
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    header = "from __future__ import annotations\n" if postponed else ""
    exec(header + _PRELUDE + textwrap.dedent(source), module.__dict__)  # noqa: S102 - the declaration under test
    return module


@pytest.mark.parametrize(
    ("source", "members", "required", "extra_items", "declared"),
    [
        (
            """
                class T(TypedDict, closed=True):
                    a: int
                    b: NotRequired[int]
                    c: ReadOnly[NotRequired[str]]
                    d: Annotated[NotRequired[int], 'meta']
            """,
            {"a", "b", "c", "d"},
            {"a"},
            Never,
            True,
        ),
        (
            """
                class T(TypedDict, total=False, closed=True):
                    a: int
                    b: Required[int]
                    c: Annotated[Required[int], 'meta']
            """,
            {"a", "b", "c"},
            {"b", "c"},
            Never,
            True,
        ),
        (
            """
                class Base(TypedDict, total=False):
                    a: int
                    b: Required[int]
                class T(Base, closed=True):
                    c: int
                    d: NotRequired[int]
            """,
            {"a", "b", "c", "d"},
            {"b", "c"},
            Never,
            True,
        ),
        (
            """
                class Base(TypedDict):
                    a: ReadOnly[NotRequired[int]]
                class T(Base):
                    a: ReadOnly[int]
            """,
            {"a"},
            {"a"},
            object,
            False,
        ),
        (
            """
                class T(typing.TypedDict):
                    a: int
                    b: typing.NotRequired[int]
            """,
            {"a", "b"},
            {"a"},
            object,
            False,
        ),
        ("class T(TypedDict):\n    a: int\n", {"a"}, {"a"}, object, False),
        ("class T(TypedDict, closed=False):\n    a: int\n", {"a"}, {"a"}, object, True),
        ("class T(TypedDict, extra_items=int):\n    a: int\n", {"a"}, {"a"}, int, True),
        ("class T(TypedDict, extra_items='int'):\n    a: int\n", {"a"}, {"a"}, int, True),
        (
            """
                class T(TypedDict, extra_items=ReadOnly[str]):
                    a: int
            """,
            {"a"},
            {"a"},
            str,
            True,
        ),
        (
            """
                class Base(TypedDict, closed=True):
                    a: int
                class T(Base):
                    pass
            """,
            {"a"},
            {"a"},
            Never,
            True,
        ),
        (
            """
                class Base(TypedDict, extra_items=ReadOnly[int]):
                    a: int
                class T(Base):
                    b: NotRequired[int]
            """,
            {"a", "b"},
            {"a"},
            int,
            True,
        ),
        (
            """
                class Base(TypedDict):
                    a: int
                class T(Base, closed=True):
                    pass
            """,
            {"a"},
            {"a"},
            Never,
            True,
        ),
        (
            """
                class Base(TypedDict, closed=False):
                    a: int
                class T(Base):
                    pass
            """,
            {"a"},
            {"a"},
            object,
            True,
        ),
        (
            """
                class G(TypedDict, Generic[V], closed=True):
                    a: int
                class T(G[int]):
                    pass
            """,
            {"a"},
            {"a"},
            Never,
            True,
        ),
        (
            """
                class A(TypedDict, closed=True):
                    a: int
                class B(TypedDict, closed=True):
                    b: NotRequired[int]
                class T(A, B):
                    pass
            """,
            {"a", "b"},
            {"a"},
            Never,
            True,
        ),
    ],
    ids=[
        "qualifiers-in-any-order",
        "total-false-and-required",
        "totality-is-per-class",
        "read-only-key-required-by-a-subclass",
        "stdlib-typeddict",
        "open-by-default",
        "open-when-said",
        "extra-items",
        "extra-items-written-as-a-string",
        "extra-items-read-only",
        "closed-is-inherited",
        "extra-items-are-inherited",
        "a-subclass-closes-an-open-base",
        "open-said-by-a-base",
        "through-a-generic-base",
        "bases-that-agree",
    ],
)
@pytest.mark.parametrize("postponed", [False, True], ids=["evaluated", "postponed"])
def test_typeddict_keys_follow_the_typing_spec(
    source: str,
    members: set[str],
    required: set[str],
    extra_items: object,
    declared: bool,
    postponed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Written as strings, as `from __future__ import annotations` writes
    # every annotation, a qualifier is invisible to `__required_keys__`;
    # read off the evaluated annotations it is not, so both spellings of
    # a declaration read alike. Openness is inherited, which the runtime
    # does not record.
    keys = typeddict_keys(_declared(source, postponed=postponed, monkeypatch=monkeypatch).T)
    assert set(keys.members) == members
    assert keys.required == required
    assert keys.extra_items is extra_items
    assert keys.declared is declared


def test_error_a_typeddict_whose_bases_disagree_on_other_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _declared(
        """
            class A(TypedDict, closed=True):
                a: int
            class B(TypedDict, extra_items=int):
                b: int
            class T(A, B):
                pass
        """,
        postponed=False,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(TypeError, match="T: its bases disagree"):
        typeddict_keys(module.T)


@pytest.mark.parametrize("name", ["T", "Outer"])
def test_error_a_typeddict_whose_annotations_do_not_resolve(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Named by the TypedDict that holds the annotation, at any depth: a
    # parser compiled around it cannot be.
    module = _declared(
        """
            class T(TypedDict):
                a: Missing
            class Outer(TypedDict):
                inner: T
        """,
        postponed=True,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(TypeError, match="T: name 'Missing' is not defined"):
        parser_for(getattr(module, name), no_leaf)


T = TypeVar("T")


class Generic_(TypedDict, Generic[T], closed=True):
    x: T


@pytest.mark.parametrize(
    "annotation",
    [set[int], list[int], Literal[b"x"], Generic_, Generic_[int]],
    ids=["set", "list", "literal-bytes", "generic-typeddict", "generic-alias"],
)
def test_error_an_annotation_outside_the_shapes_has_no_parser(annotation: object) -> None:
    assert parser_for(annotation, no_leaf) is None
    with pytest.raises(TypeError, match="is not a shape JSON takes"):
        parser(annotation, no_leaf)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the `type` statement is 3.12 syntax")
def test_a_type_statement_alias_may_hold_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _declared(
        "type Nested = int | tuple[Nested, ...]\n", postponed=False, monkeypatch=monkeypatch
    )
    assert parser(module.Nested, no_leaf)([1, [2, [3]]], ()) == ((1, (2, (3,))), ())


def test_a_leaf_is_asked_first_at_every_depth() -> None:
    # The one hook: a caller's own shapes, consulted before the built-in
    # ones, inside an array or an object as at the top.
    class Marker:
        pass

    def leaf(annotation: object) -> Parser | None:
        if annotation is not Marker:
            return None

        def parse(value: object, loc: Loc) -> Parsed:
            if isinstance(value, str) and value.startswith("m:"):
                return value[2:], ()
            return value, ((loc, "expected a marker"),)  # pyright: ignore[reportReturnType]

        return parse

    read = parser(tuple[Marker, ...], leaf)
    assert read(["m:a", "m:b"], ()) == (("a", "b"), ())


@pytest.mark.parametrize(
    ("annotation", "shape"),
    [
        (Literal[True], "bool"),
        (Literal[1, 2], "int"),
        (Literal["a", "b"], "str"),
        (Literal[0, "auto"], None),
        (None, "null"),
        (tuple[int, ...], "tuple"),
        (Options, "mapping"),
        (Width, "int"),
        (JSONValue, None),
    ],
)
def test_a_shape_is_what_a_union_dispatches_on(annotation: object, shape: str | None) -> None:
    assert shape_of(annotation) == shape


def test_a_mixed_literal_is_described() -> None:
    assert describe(Literal[0, "auto"]) == "one of ('auto', 0)"
    assert describe(int | None) == "an integer or null"
