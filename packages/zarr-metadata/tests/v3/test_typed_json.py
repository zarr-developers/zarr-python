"""The checker: parsers and writers compiled from annotations for the shapes JSON takes.

Tested on its own, with no entity in sight. A leaf stands in for
whatever a caller adds -- the entity layer adds the shape of a field
holding another entity -- and `no_leaf` adds nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, NewType, NotRequired, cast

import pytest
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.model import UNSET
from zarr_metadata.v3._typed_json import (
    Loc,
    Parsed,
    Parser,
    declared_class_vars,
    describe,
    field_hints,
    is_class_var,
    is_not_required,
    is_optional,
    no_leaf,
    no_writer_leaf,
    parser,
    parser_for,
    record_writer,
    shape_of,
    writer,
    writer_for,
)

Level = NewType("Level", int)


class Options(TypedDict, closed=True):
    level: int
    note: NotRequired[str]
    tag: ReadOnly[NotRequired[str]]


@dataclass(frozen=True)
class Record:
    level: int
    note: str | UNSET = UNSET


@dataclass(frozen=True)
class Checked:
    level: int

    def __post_init__(self) -> None:
        return None


@dataclass(frozen=True)
class Declared:
    level: int
    identifier: ClassVar[str] = "declared"
    owed: ClassVar[int]


def _read(annotation: object, value: object) -> Parsed:
    return parser(annotation, no_leaf)(value, (), None)


@pytest.mark.parametrize(
    ("annotation", "value", "parsed"),
    [
        (int, 3, 3),
        (float, 3, 3),
        (float, 2.5, 2.5),
        (bool, True, True),
        (str, "x", "x"),
        (Literal["a", "b"], "a", "a"),
        (Literal[1, 2], 2, 2),
        (tuple[int, ...], [1, 2], (1, 2)),
        (tuple[int, str], (1, "a"), (1, "a")),
        (int | str, "a", "a"),
        (Mapping[str, int], {"k": 1}, {"k": 1}),
        (JSONValue, {"any": [1, None]}, {"any": [1, None]}),
        (Level, 5, 5),
        (int | UNSET, 4, 4),
    ],
    ids=[
        "int",
        "number-from-int",
        "number",
        "bool",
        "str",
        "literal-str",
        "literal-int",
        "sequence",
        "fixed-tuple",
        "union",
        "mapping",
        "json",
        "newtype",
        "optional-present",
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


def test_a_typed_dict_keeps_what_the_document_said() -> None:
    # A closed TypedDict member reports a key it does not declare and
    # keeps it: the member still says what the document said. An
    # optional key left out stays out.
    read = parser(Options, no_leaf)
    assert read({"level": 1}, (), None) == ({"level": 1}, ())
    value, problems = read({"level": 1, "extra": 2}, (), None)
    assert value == {"level": 1, "extra": 2}
    assert [(problem.loc, problem.kind) for problem in problems] == [(("extra",), "unknown_key")]


def test_error_a_typed_dict_misses_a_required_key() -> None:
    value, problems = parser(Options, no_leaf)({"note": "n"}, (), None)
    assert value == {"note": "n"}
    assert [(problem.loc, problem.kind) for problem in problems] == [(("level",), "missing_key")]


def test_a_record_is_built_when_every_member_reads() -> None:
    # An optional member left out is `UNSET` in the record, so no
    # default of the record's own decides what a document said; an
    # unknown key is reported and survives.
    read = parser(Record, no_leaf)
    assert read({"level": 1}, (), None) == (Record(level=1, note=UNSET), ())
    record, problems = read({"level": 1, "typo": 0}, (), None)
    assert record == Record(level=1)
    assert [(problem.loc, problem.kind) for problem in problems] == [(("typo",), "unknown_key")]


def test_error_a_record_is_not_built_around_a_member_that_did_not_read() -> None:
    value, problems = parser(Record, no_leaf)({"level": "high"}, (), None)
    assert value == {"level": "high"}
    assert [(problem.loc, problem.kind, problem.message) for problem in problems] == [
        (("level",), "invalid_type", "expected an integer, got 'high'")
    ]


def test_error_a_record_may_not_define_post_init() -> None:
    # A record is plain data, built whenever its keys read; a rule about
    # it belongs where the caller asks for rules.
    with pytest.raises(TypeError, match="Checked defines __post_init__; a record is plain data"):
        parser(Checked, no_leaf)


def test_error_an_annotation_outside_the_shapes_has_no_parser() -> None:
    assert parser_for(set[int], no_leaf) is None
    with pytest.raises(TypeError, match="is not a shape JSON takes"):
        parser(set[int], no_leaf)


def test_a_leaf_is_asked_first_at_every_depth() -> None:
    # The one hook: a caller's own shapes, consulted before the built-in
    # ones, inside an array or an object as at the top.
    class Marker:
        pass

    def leaf(annotation: object) -> Parser[None] | None:
        if annotation is not Marker:
            return None

        def parse(value: object, loc: Loc, state: None) -> Parsed:
            if isinstance(value, str) and value.startswith("m:"):
                return value[2:], ()
            return value, ((loc, "expected a marker"),)  # pyright: ignore[reportReturnType]

        return parse

    read = parser(tuple[Marker, ...], leaf)
    assert read(["m:a", "m:b"], (), None) == (("a", "b"), ())


def test_writers_are_the_parsers_inverse() -> None:
    # What a parser reads from a document, a writer puts back: a record
    # as the object of its present members, an absent optional member
    # left out, a JSON-valued member copied rather than shared.
    assert record_writer(Record, no_writer_leaf)(Record(level=1)) == {"level": 1}
    assert record_writer(Record, no_writer_leaf)(Record(level=1, note="n")) == {
        "level": 1,
        "note": "n",
    }
    assert writer(tuple[int, str], no_writer_leaf)((1, "a")) == (1, "a")
    held = {"k": [1, 2]}
    written = writer(JSONValue, no_writer_leaf)(held)
    assert written == held
    assert written is not held


def test_error_a_fixed_tuple_of_the_wrong_length_is_not_written() -> None:
    write = writer_for(tuple[int, str], no_writer_leaf)
    assert write is not None
    with pytest.raises(TypeError, match="is not a JSON value"):
        write((1,))


def test_field_hints_resolve_per_class_and_skip_class_variables() -> None:
    # Under PEP 649 the annotations are strings, resolved where the class
    # is; a class variable is not a field, and one a base annotates and
    # nothing sets is owed.
    assert dict(field_hints(Declared)) == {"level": int}
    assert declared_class_vars(Declared) == {"identifier": Declared, "owed": Declared}
    hints = field_hints(Declared)
    assert field_hints(Declared) is hints
    with pytest.raises(TypeError, match="does not support item assignment"):
        cast("dict[str, object]", hints)["level"] = str


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("ClassVar[str]", True),
        ("ClassVar", True),
        ("typing.ClassVar[str]", True),
        ("t.ClassVar[str]", True),
        ("str", False),
        ("ClassVarLike[str]", False),
        ("Final[ClassVar[str]]", False),
    ],
)
def test_a_class_var_is_read_from_any_spelling(annotation: str, expected: bool) -> None:
    assert is_class_var(annotation) is expected


@pytest.mark.parametrize(
    ("annotation", "shape"),
    [
        (Literal[True], "bool"),
        (Literal[1, 2], "int"),
        (Literal["a", "b"], "str"),
        (Literal[0, "auto"], None),
        (tuple[int, ...], "tuple"),
        (Options, "mapping"),
        (JSONValue, None),
    ],
)
def test_a_shape_is_what_a_union_dispatches_on(annotation: object, shape: str | None) -> None:
    assert shape_of(annotation) == shape


def test_a_mixed_literal_is_described() -> None:
    assert describe(Literal[0, "auto"]) == "one of ('auto', 0)"


def test_optional_and_not_required_are_read_off_the_annotation() -> None:
    assert is_optional(int | UNSET)
    assert not is_optional(int)
    assert is_not_required(NotRequired[int])
    assert not is_not_required(int)
