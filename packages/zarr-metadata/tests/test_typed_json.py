"""The checker: parsers compiled from annotations for the shapes JSON takes.

Tested on its own, with no metadata field in sight. A leaf stands in for
whatever a caller adds -- the definition layer adds the shape of a member
holding another metadata field -- and `no_leaf` adds nothing.
"""

from __future__ import annotations

import importlib
import itertools
import math
import pkgutil
import sys
import textwrap
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Generic, Literal, NewType, NotRequired, TypeVar, cast

import pytest
from typing_extensions import ReadOnly, TypeAliasType, TypedDict, is_typeddict

import zarr_metadata.v2
import zarr_metadata.v3
from zarr_metadata._common import JSONValue
from zarr_metadata._typed_json import (
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
from zarr_metadata.model import ZarrV2ArrayMetadata, ZarrV3ArrayMetadata
from zarr_metadata.typed_json import check
from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON, ZarrV2DataTypeMetadata
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

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


class GzipLevel(TypedDict, closed=True):
    level: int


class Gzip(TypedDict, closed=True):
    name: Literal["gzip"]
    configuration: GzipLevel


class Blosc(TypedDict, closed=True):
    name: Literal["blosc"]
    configuration: Options


class Pair(TypedDict, closed=True):
    a: int
    b: int


class Triple(TypedDict, closed=True):
    a: str
    b: str
    c: str


class Option1(TypedDict, closed=False):
    one: NotRequired[int]


class Option2(TypedDict, closed=False):
    two: NotRequired[tuple[int, ...]]


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
        (
            Blosc | Gzip,
            {"name": "gzip", "configuration": {"level": "x"}},
            ("configuration", "level"),
            "invalid_type",
        ),
        (Blosc | Gzip, {"name": "zstd", "configuration": {}}, ("name",), "invalid_value"),
        (Blosc | Gzip, {"configuration": {}}, ("name",), "missing_key"),
        (Triple | Pair, {"a": 1, "b": "x"}, ("b",), "invalid_type"),
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
        "a-tag-picks-the-branch-that-reports",
        "a-tag-no-branch-has",
        "a-tag-missing",
        "untagged-the-closest-branch-reports",
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
        (Option1 | Option2, {"two": [1]}, {"two": (1,)}, []),
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
        "union-takes-the-branch-declaring-the-most-keys",
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
from typing import Annotated, Generic, NotRequired, Required, TypeVar
from typing_extensions import Never, ReadOnly
from {typeddicts} import TypedDict
V = TypeVar("V")
"""

_modules = itertools.count()

_TYPING_TAKES_PEP_728 = sys.version_info >= (3, 15)
"""Whether `typing.TypedDict` takes `closed=` and `extra_items=`, as `typing_extensions` does."""


def _declared(
    source: str,
    *,
    postponed: bool,
    monkeypatch: pytest.MonkeyPatch,
    typeddicts: str = "typing_extensions",
) -> types.ModuleType:
    """The module `source` defines, written with or without `from __future__ import annotations`.

    Registered in `sys.modules`, because a string annotation is resolved
    in the module its class was defined in. `typeddicts` is the module the
    `TypedDict` it declares with comes from.
    """
    name = f"tests._typeddicts_{next(_modules)}"
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    header = "from __future__ import annotations\n" if postponed else ""
    text = header + _PRELUDE.format(typeddicts=typeddicts) + textwrap.dedent(source)
    exec(text, module.__dict__)  # noqa: S102 - the declaration under test
    return module


_DECLARATIONS = [
    pytest.param(
        """
            class T(TypedDict, closed=True):
                a: int
                b: NotRequired[int]
                c: ReadOnly[NotRequired[str]]
                d: Annotated[NotRequired[int], 'meta']
        """,
        {"a", "b", "c", "d"},
        {"a"},
        "Never",
        True,
        id="qualifiers-in-any-order",
    ),
    pytest.param(
        """
            class T(TypedDict, total=False, closed=True):
                a: int
                b: Required[int]
                c: Annotated[Required[int], 'meta']
        """,
        {"a", "b", "c"},
        {"b", "c"},
        "Never",
        True,
        id="total-false-and-required",
    ),
    pytest.param(
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
        "Never",
        True,
        id="totality-is-per-class",
    ),
    pytest.param(
        """
            class Base(TypedDict):
                a: ReadOnly[NotRequired[int]]
            class T(Base):
                a: ReadOnly[int]
        """,
        {"a"},
        {"a"},
        "object",
        False,
        id="read-only-key-required-by-a-subclass",
    ),
    pytest.param(
        """
            class Base(TypedDict):
                a: ReadOnly[int | str]
                b: ReadOnly[NotRequired[int]]
            class T(Base):
                a: ReadOnly[int]
                b: ReadOnly[Required[int]]
        """,
        {"a", "b"},
        {"a", "b"},
        "object",
        False,
        id="read-only-keys-narrowed",
    ),
    pytest.param(
        """
            class Base(TypedDict, total=False):
                a: int
            class T(Base):
                a: int
        """,
        {"a"},
        {"a"},
        "object",
        False,
        id="redeclared-by-a-total-subclass",
    ),
    pytest.param(
        """
            class T(TypedDict, total=False):
                a: Annotated['Annotated[Required[str], "x"]', "y"]
                b: str
        """,
        {"a", "b"},
        {"a"},
        "object",
        False,
        id="a-qualifier-quoted-inside-annotated",
    ),
    pytest.param(
        """
            T = TypedDict("T", {"class": int, "that's": NotRequired[str], "": ReadOnly[bool]})
        """,
        {"class", "that's", ""},
        {"class", ""},
        "object",
        False,
        id="functional-syntax-with-keys-that-are-not-names",
    ),
    pytest.param(
        "class T(TypedDict):\n    a: int\n",
        {"a"},
        {"a"},
        "object",
        False,
        id="open-by-default",
    ),
    pytest.param(
        "class T(TypedDict, closed=False):\n    a: int\n",
        {"a"},
        {"a"},
        "object",
        True,
        id="open-when-said",
    ),
    pytest.param(
        "class T(TypedDict, extra_items=int):\n    a: int\n",
        {"a"},
        {"a"},
        "int",
        True,
        id="extra-items",
    ),
    pytest.param(
        "class T(TypedDict, extra_items='int'):\n    a: int\n",
        {"a"},
        {"a"},
        "int",
        True,
        id="extra-items-written-as-a-string",
    ),
    pytest.param(
        "class T(TypedDict, extra_items='tuple[T, ...]'):\n    a: int\n",
        {"a"},
        {"a"},
        "tuple[T, ...]",
        True,
        id="extra-items-naming-its-own-class",
    ),
    pytest.param(
        "class T(TypedDict, extra_items=ReadOnly[str]):\n    a: int\n",
        {"a"},
        {"a"},
        "str",
        True,
        id="extra-items-read-only",
    ),
    pytest.param(
        """
            class Base(TypedDict, closed=True):
                a: int
            class T(Base):
                pass
        """,
        {"a"},
        {"a"},
        "Never",
        True,
        id="closed-is-inherited",
    ),
    pytest.param(
        """
            class Base(TypedDict, extra_items=ReadOnly[int]):
                a: int
            class T(Base):
                b: NotRequired[int]
        """,
        {"a", "b"},
        {"a"},
        "int",
        True,
        id="extra-items-are-inherited",
    ),
    pytest.param(
        """
            class Base(TypedDict, extra_items=int):
                a: int
            class Left(Base):
                pass
            class Right(Base):
                pass
            class T(Left, Right):
                pass
        """,
        {"a"},
        {"a"},
        "int",
        True,
        id="a-diamond-over-extra-items",
    ),
    pytest.param(
        """
            class Base(TypedDict):
                a: int
            class T(Base, closed=True):
                pass
        """,
        {"a"},
        {"a"},
        "Never",
        True,
        id="a-subclass-closes-an-open-base",
    ),
    pytest.param(
        """
            class Base(TypedDict, closed=False):
                a: int
            class T(Base):
                pass
        """,
        {"a"},
        {"a"},
        "object",
        True,
        id="open-said-by-a-base",
    ),
    pytest.param(
        """
            class G(TypedDict, Generic[V], closed=True):
                a: int
            class T(G[int]):
                pass
        """,
        {"a"},
        {"a"},
        "Never",
        True,
        id="through-a-generic-base",
    ),
    pytest.param(
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
        "Never",
        True,
        id="bases-that-agree",
    ),
    pytest.param(
        """
            class A(TypedDict, closed=False):
                a: int
            class B(TypedDict, closed=True):
                b: int
            class T(A, B):
                pass
        """,
        {"a", "b"},
        {"a", "b"},
        "Never",
        True,
        id="a-closed-base-beside-an-open-one",
    ),
]


def _under_both_typeddicts(declarations: list[ParameterSet]) -> list[ParameterSet]:
    """Each declaration with `typing_extensions.TypedDict`, and with `typing.TypedDict` where it can say it."""
    both: list[ParameterSet] = []
    for declaration in declarations:
        source = cast("str", declaration.values[0])
        both.append(pytest.param(*declaration.values, "typing_extensions", id=f"{declaration.id}"))
        if _TYPING_TAKES_PEP_728 or ("closed=" not in source and "extra_items=" not in source):
            both.append(pytest.param(*declaration.values, "typing", id=f"{declaration.id}-typing"))
    return both


@pytest.mark.parametrize(
    ("source", "members", "required", "extra_items", "declared", "typeddicts"),
    _under_both_typeddicts(_DECLARATIONS),
)
@pytest.mark.parametrize("postponed", [False, True], ids=["evaluated", "postponed"])
def test_typeddict_keys_follow_the_typing_spec(
    source: str,
    members: set[str],
    required: set[str],
    extra_items: str,
    declared: bool,
    typeddicts: str,
    postponed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Written as strings, as `from __future__ import annotations` writes
    # every annotation, a qualifier is invisible to `__required_keys__`,
    # as it is to `typing.TypedDict` on 3.11 under `ReadOnly` or
    # `Annotated`; read off the evaluated annotations it is not, so every
    # spelling of a declaration reads alike. Openness is inherited, which
    # the runtime does not record.
    module = _declared(source, postponed=postponed, monkeypatch=monkeypatch, typeddicts=typeddicts)
    keys = typeddict_keys(module.T)
    assert set(keys.members) == members
    assert keys.required == required
    assert keys.extra_items == eval(extra_items, vars(module))
    assert keys.declared is declared


_WRITTEN_WHERE = """
    class _Hidden(TypedDict):
        x: int
    class Base(TypedDict, total=False):
        nested: tuple['_Hidden', ...]
        top: '_Hidden'
        kept: ReadOnly[int | str]
"""


@pytest.mark.parametrize(
    "child_postponed", [False, True], ids=["child-evaluated", "child-postponed"]
)
@pytest.mark.parametrize("base_postponed", [False, True], ids=["base-evaluated", "base-postponed"])
def test_a_key_reads_where_its_class_was_written(
    base_postponed: bool, child_postponed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The subclass's module never imports `_Hidden`, and the runtime ties a
    # string to its module only at the top of an annotation; each key still
    # reads in the module of the class that declared it.
    base = _declared(_WRITTEN_WHERE, postponed=base_postponed, monkeypatch=monkeypatch)
    child = _declared(
        f"from {base.__name__} import Base\nclass T(Base):\n    own: int\n    kept: ReadOnly[int]\n",
        postponed=child_postponed,
        monkeypatch=monkeypatch,
    )
    assert typeddict_keys(child.T).required == {"kept", "own"}
    value = {"nested": [{"x": 1}], "top": {"x": 2}, "kept": 3, "own": 4}
    assert parser(child.T, no_leaf)(value, ()) == ({**value, "nested": ({"x": 1},)}, ())


@pytest.mark.parametrize(
    "postponed",
    [
        pytest.param(
            False,
            id="evaluated",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 14), reason="an unquoted later name needs PEP 649"
            ),
        ),
        pytest.param(True, id="postponed"),
    ],
)
def test_a_typeddict_may_name_a_class_written_after_it(
    postponed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _declared(
        """
            class T(TypedDict):
                later: NotRequired[Later]
            class Later(TypedDict):
                x: int
        """,
        postponed=postponed,
        monkeypatch=monkeypatch,
    )
    assert parser(module.T, no_leaf)({"later": {"x": 1}}, ()) == ({"later": {"x": 1}}, ())


@pytest.mark.parametrize(
    ("base_source", "postponed"),
    [
        ("Foo = int\nclass Base(TypedDict):\n    x: Foo\n    items: tuple[Foo, ...]\n", True),
        ("Foo = int\nclass Base(TypedDict):\n    x: 'Foo'\n    items: tuple['Foo', ...]\n", False),
    ],
    ids=["postponed", "quoted-by-hand"],
)
def test_a_name_reads_as_the_module_that_wrote_it_means_it(
    base_source: str, postponed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The subclass's module means something else by `Foo`; the keys still
    # read as their base wrote them, at the top of an annotation or inside.
    base = _declared(base_source, postponed=postponed, monkeypatch=monkeypatch)
    child = _declared(
        f"from {base.__name__} import Base\nFoo = str\nclass T(Base):\n    pass\n",
        postponed=True,
        monkeypatch=monkeypatch,
    )
    read = parser(child.T, no_leaf)
    assert read({"x": 1, "items": [2]}, ())[1] == ()
    assert [(found.loc, found.kind) for found in read({"x": "s", "items": ["t"]}, ())[1]] == [
        (("x",), "invalid_type"),
        (("items", 0), "invalid_type"),
    ]


def test_error_a_key_both_required_and_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    # The spec allows one; the runtime and other checkers disagree on which wins.
    module = _declared(
        "class T(TypedDict):\n    a: Required[NotRequired[int]]\n",
        postponed=True,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(TypeError, match="T.a: Required and NotRequired both"):
        typeddict_keys(module.T)


def test_error_a_typeddict_whose_runtime_keys_are_not_its_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Were they to disagree, a key would stop being required unnoticed.
    module = _declared(
        "class T(TypedDict):\n    a: int\n", postponed=False, monkeypatch=monkeypatch
    )
    monkeypatch.setattr(module.T, "__required_keys__", frozenset({"a", "b"}))
    with pytest.raises(TypeError, match="T: its annotations declare"):
        typeddict_keys(module.T)


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


def test_error_a_key_that_does_not_resolve_where_it_was_written_names_its_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Asked through a subclass elsewhere, the class that wrote it is named.
    base = _declared(
        "class Base(TypedDict):\n    items: tuple['Missing', ...]\n",
        postponed=False,
        monkeypatch=monkeypatch,
    )
    child = _declared(
        f"from {base.__name__} import Base\nclass T(Base):\n    own: int\n",
        postponed=False,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(TypeError, match="Base: name 'Missing' is not defined"):
        typeddict_keys(child.T)


def test_error_a_typeddict_written_in_a_function_resolves_only_in_its_module() -> None:
    # This module postpones annotations, and `Local` is no name of it.
    class Local(TypedDict, closed=True):
        x: int

    class Holder(TypedDict, closed=True):
        local: Local

    with pytest.raises(TypeError, match="Holder: name 'Local' is not defined"):
        typeddict_keys(Holder)


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


def test_a_shape_is_described_as_a_message_would_name_it() -> None:
    assert describe(Literal[0, "auto"]) == "one of ('auto', 0)"
    assert describe(int | None) == "an integer or null"
    assert describe(Width) == "an integer"
    assert describe(ZarrV2DataTypeMetadata) == "a ZarrV2DataTypeMetadata"


# --- check: the public door ------------------------------------------------

_V3_DOCUMENT = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
_V2_DOCUMENT = {
    **ZarrV2ArrayMetadata.create_default(shape=(4,)).to_json(),
    "dtype": [["a", "<f8"], ["b", [["c", "|u1", [2]]]]],
}


class Unreadable(TypedDict, closed=True):
    inner: Options
    members: set[int]


class Holds(TypedDict, closed=True):
    unreadable: Unreadable


@pytest.mark.parametrize(
    ("value", "shape", "found"),
    [
        (_V3_DOCUMENT, ZarrV3ArrayMetadataJSON, []),
        ({**_V3_DOCUMENT, "dimension_names": [None]}, ZarrV3ArrayMetadataJSON, []),
        ({**_V3_DOCUMENT, "acme": {"must_understand": False}}, ZarrV3ArrayMetadataJSON, []),
        (_V2_DOCUMENT, ZarrV2ArrayMetadataJSON, []),
        ({"level": 1, "extra": 2}, Options, [(("extra",), "unknown_key")]),
    ],
    ids=["v3-array", "null-in-a-union", "extra-items", "v2-structured-dtype", "unknown-key"],
)
def test_check_gives_a_value_of_the_typeddict(
    value: dict[str, object], shape: type, found: list[tuple[Loc, str]]
) -> None:
    # Any TypedDict the package declares, and any other: a value of it,
    # arrays as tuples, holding what it admits and nothing else.
    typed, problems = check(value, shape)
    assert [(problem.loc, problem.kind) for problem in problems] == found
    assert typed is not None
    assert parser(shape, no_leaf)(typed, ())[1] == ()
    assert set(typed) == {key for key in value if (key,) not in {loc for loc, _ in found}}


def test_error_check_is_asked_of_a_typeddict() -> None:
    with pytest.raises(TypeError, match="is not a TypedDict"):
        check({}, dict)


def test_error_check_names_what_no_parser_reads() -> None:
    # Down to the TypedDict that holds it, at any depth.
    with pytest.raises(TypeError, match=r"Holds.unreadable: Unreadable: members is not a shape"):
        check({}, Holds)


@pytest.mark.parametrize(
    ("value", "loc", "kind"),
    [
        ({"level": math.nan}, ("level",), "invalid_value"),
        ({"level": 1, "note": math.inf}, ("note",), "invalid_value"),
        ({"level": 1, 2: 3}, (), "invalid_type"),
        ({"level": 1, "note": {1}}, ("note",), "invalid_type"),
        ({"level": 1, "note": [{3: 1}]}, ("note", 0), "invalid_type"),
    ],
    ids=[
        "nan",
        "infinity",
        "a-key-that-is-not-a-string",
        "a-set",
        "nested-key-that-is-not-a-string",
    ],
)
def test_error_check_locates_a_value_that_is_not_json(
    value: dict[object, object], loc: Loc, kind: str
) -> None:
    # Refused before any TypedDict is asked: what `check` compares is JSON.
    typed, problems = check(value, Options)
    assert typed is None
    assert [(problem.loc, problem.kind) for problem in problems] == [(loc, kind)]


def test_error_check_refuses_a_value_of_another_type() -> None:
    typed, problems = check({"level": "high"}, Options)
    assert typed is None
    assert [(problem.loc, problem.kind) for problem in problems] == [(("level",), "invalid_type")]


def _declared_typeddicts() -> list[type]:
    """Every TypedDict a public module of the package's JSON types exports."""
    found: dict[type, None] = {}
    for package in (zarr_metadata.v2, zarr_metadata.v3):
        modules = [package.__name__] + [
            info.name
            for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}.")
            if not any(part.startswith("_") for part in info.name.split("."))
        ]
        for name in modules:
            module = importlib.import_module(name)
            for exported in getattr(module, "__all__", ()):
                value = getattr(module, exported)
                if isinstance(value, type) and is_typeddict(value):
                    found[value] = None
    return list(found)


@pytest.mark.parametrize(
    "typeddict", _declared_typeddicts(), ids=lambda typeddict: typeddict.__name__
)
def test_every_typeddict_the_package_declares_compiles(typeddict: type) -> None:
    # A declaration no parser reads would be a document type `check` could
    # not be asked about.
    assert parser_for(typeddict, no_leaf) is not None
