"""What a problem carries besides its message: what was found, and what was expected.

As pydantic's errors and zod's issues do, a problem carries its message's
data: `input`, what the value handed in holds at the problem's `loc`, and
`ctx`, what was expected there, where that is more than a type -- a
type's bounds, a closed set's values. Every reader fills `input`, so a
rule says only where a problem is.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import pytest
from annotated_types import Le
from typing_extensions import TypedDict

from zarr_metadata._json import (
    arrays_to_tuples,
    is_canonical_json,
    validate_json,
    value_at,
    with_input,
    within,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model import (
    MetadataValidationError,
    ValidationProblem,
    ZarrV2ConsolidatedMetadata,
    ZarrV3ArrayMetadata,
    parse_array_metadata_v3,
    read_array_metadata_v3,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
    validate_metadata_field_v3,
    validate_node_metadata_v3,
    validate_node_name_v3,
    validate_node_path_v3,
)
from zarr_metadata.typed_json import (
    check,
)
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    CodecDefinition,
    DataTypeDefinition,
    fill_value_problems,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from zarr_metadata._common import JSONValue


@pytest.mark.parametrize(
    ("found", "ctx"),
    [
        (UNSET, {}),
        (12, {"ge": 0, "le": 9}),
        (None, {"expected": ["C", "F"]}),
    ],
    ids=["nothing-found", "a-bound", "a-null-found"],
)
def test_a_problem_carries_what_was_found_and_what_was_expected(
    found: JSONValue | UNSET, ctx: dict[str, JSONValue]
) -> None:
    written = dict(ctx)
    problem = ValidationProblem(("level",), "bad level", "invalid_value", input=found, ctx=written)
    assert problem.input is found
    # Held as a read-only view of a copy, arrays as tuples.
    assert dict(problem.ctx) == arrays_to_tuples(ctx)
    # What it says, and where, is the problem; what it carries is detail.
    bare = ValidationProblem(("level",), "bad level", "invalid_value")
    assert problem == bare
    assert hash(problem) == hash(bare)
    assert repr(problem) == repr(bare)
    assert str(problem) == "level: bad level"
    # A raised error is a finished report.
    written["more"] = 1
    assert dict(problem.ctx) == arrays_to_tuples(ctx)
    with pytest.raises(TypeError):
        problem.ctx["ge"] = 1  # pyright: ignore[reportIndexIssue]


@pytest.mark.parametrize(
    "ctx",
    [["ge", 0], {1: 0}, {"ge": {0}}, {"ge": math.nan}],
    ids=["an-array", "a-key-that-is-not-a-string", "a-set", "nan"],
)
def test_error_a_problem_refuses_a_ctx_that_is_not_an_object_of_json_values(ctx: object) -> None:
    with pytest.raises(TypeError, match="ctx is an object of JSON values"):
        ValidationProblem(("level",), "bad level", "invalid_value", ctx=ctx)  # pyright: ignore[reportArgumentType]


def test_an_error_about_what_is_not_json_pickles() -> None:
    # A problem holds no input that is not JSON, which might not pickle,
    # so the error it is raised in crosses to another process as it is.
    value = {"shape": [lambda: 4]}
    problems = validate_json(value)
    assert [(problem.loc, problem.input) for problem in problems] == [(("shape", 0), UNSET)]
    again = pickle.loads(pickle.dumps(MetadataValidationError(problems)))
    assert again.problems == problems


def test_a_problem_pickles_and_copies_with_what_it_carries() -> None:
    problem = ValidationProblem(("shape",), "bad", "invalid_value", input=[12], ctx={"ge": 0})
    error = MetadataValidationError([problem])
    for again in (
        pickle.loads(pickle.dumps(problem)),
        copy.copy(problem),
        copy.deepcopy(problem),
        pickle.loads(pickle.dumps(error)).problems[0],
    ):
        assert again == problem
        assert again.input == [12]
        assert dict(again.ctx) == {"ge": 0}
        with pytest.raises(TypeError):
            again.ctx["ge"] = 1  # pyright: ignore[reportIndexIssue]


def _deep(levels: int) -> list[object]:
    """An array nested `levels` deep, deeper than the interpreter walks."""
    value: list[object] = []
    for _ in range(levels):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("value", "at", "loc", "held", "found"),
    [
        ({"a": [1, {"b": 2}]}, (), ("a", 1, "b"), UNSET, 2),
        ({"a": None}, (), ("a",), UNSET, None),
        ({"a": 1}, (), ("b",), UNSET, UNSET),
        ({"a": [1]}, (), ("a", 1), UNSET, UNSET),
        ({"a": "text"}, (), ("a", 0), UNSET, UNSET),
        ({"a": [1]}, (), ("a", "b"), UNSET, UNSET),
        ([1, 2], ("codecs",), ("codecs", 1), UNSET, 2),
        ([1, 2], ("codecs",), ("storage_transformers", 1), UNSET, UNSET),
        ({"a": {1, 2}}, (), ("a",), UNSET, UNSET),
        ({"a": _deep(100_000)}, (), ("a",), UNSET, UNSET),
        ({"a": 1}, (), ("a",), "what a reader inside found", 1),
        ({"a": {1, 2}}, (), ("a",), [1, 2], [1, 2]),
    ],
    ids=[
        "nested",
        "a-null",
        "a-missing-key",
        "past-the-end",
        "a-string-is-no-array",
        "an-array-has-no-keys",
        "under-where-the-value-sits",
        "not-under-it",
        "what-is-not-json",
        "too-deep-to-walk",
        "the-value-handed-in-wins",
        "what-is-not-json-keeps-what-it-holds",
    ],
)
def test_a_problem_holds_what_its_value_holds_at_its_loc(
    value: object,
    at: tuple[str | int, ...],
    loc: tuple[str | int, ...],
    held: JSONValue | UNSET,
    found: JSONValue | UNSET,
) -> None:
    (problem,) = with_input([ValidationProblem(loc, "bad", "invalid_value", input=held)], value, at)
    assert problem.input == found
    assert (problem.input is UNSET) is (found is UNSET)


class GzipLevelOnly(TypedDict, closed=True):
    level: int


def _raised(read: Callable[[], object]) -> Sequence[ValidationProblem]:
    with pytest.raises(MetadataValidationError) as caught:
        read()
    return caught.value.problems


BAD_ARRAY: dict[str, object] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4, 4],
    "data_type": "uint8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 300,
    "codecs": [
        {"name": "transpose", "configuration": {"order": [0, 1, 2]}},
        {"name": "bytes"},
        {"name": "gzip", "configuration": {"level": 12, "extra": [1]}},
    ],
    "dimension_names": ["x"],
    "attributes": {},
}
"""An array document as `json.loads` gives one, arrays as lists, with a problem in each part a reader reads."""

BAD_ARRAY_LOCS = {
    ("chunk_grid", "configuration", "chunk_shape"),
    ("fill_value",),
    ("codecs", 0, "configuration", "order"),
    ("codecs", 2, "configuration", "level"),
    ("codecs", 2, "configuration", "extra"),
    ("dimension_names",),
}

READERS: list[tuple[Callable[[object], Sequence[ValidationProblem]], object]] = [
    (lambda value: check(value, GzipLevelOnly)[1], {"level": 12, "extra": [1]}),
    (lambda value: GZIP_CODEC.judge(value)[1], {"level": 12, "extra": [1]}),
    (
        lambda value: resolve(value, CodecDefinition, CORE_AND_EXTENSIONS)[1],
        {"name": "gzip", "configuration": {"level": 12}, "must_understand": "yes"},
    ),
    (
        lambda value: fill_value_problems(
            resolve("uint8", DataTypeDefinition, CORE_AND_EXTENSIONS)[0], value
        ),
        300,
    ),
    (validate_json, {"a": [math.inf]}),
    (
        validate_metadata_field_v3,
        {"name": 1, "configuration": {"a": [math.nan]}, "extra": [1]},
    ),
    (validate_node_name_v3, "__/"),
    (validate_node_path_v3, "a//"),
    (validate_array_metadata_v3, BAD_ARRAY),
    (lambda value: read_array_metadata_v3(value).problems, BAD_ARRAY),
    (lambda value: _raised(lambda: parse_array_metadata_v3(value)), BAD_ARRAY),
    (lambda value: _raised(lambda: ZarrV3ArrayMetadata.from_json(value)), BAD_ARRAY),
    (validate_node_metadata_v3, BAD_ARRAY),
    (
        validate_group_metadata_v3,
        {
            "zarr_format": 3,
            "node_type": "group",
            "consolidated_metadata": {
                "kind": "inline",
                "must_understand": [False],
                "metadata": {"a": BAD_ARRAY, "__b": BAD_ARRAY, "a/c": BAD_ARRAY, "d/e": BAD_ARRAY},
            },
        },
    ),
    (
        validate_array_metadata_v2,
        {"zarr_format": 2, "shape": [4], "chunks": [2, 2], "order": "Q", "filters": [[1]]},
    ),
    (validate_group_metadata_v2, {"zarr_format": 3, "extra": [1]}),
    (
        lambda value: _raised(lambda: ZarrV2ConsolidatedMetadata.from_json(value)),
        {"zarr_consolidated_format": [1], "metadata": {"a/.zarray": [math.inf]}},
    ),
    (
        lambda value: _raised(lambda: ZarrV3ArrayMetadata.from_key_value(value)),  # pyright: ignore[reportArgumentType]
        {"zarr.json": b"{"},
    ),
]


@pytest.mark.parametrize(
    ("read", "value"),
    READERS,
    ids=[
        "check",
        "judge",
        "resolve",
        "fill-value-problems",
        "validate-json",
        "validate-metadata-field",
        "validate-node-name",
        "validate-node-path",
        "validate-array",
        "read-array",
        "parse-array",
        "array-from-json",
        "validate-node",
        "validate-group-and-what-it-holds",
        "validate-array-v2",
        "validate-group-v2",
        "v2-consolidated-from-json",
        "from-key-value",
    ],
)
def test_every_problem_a_reader_finds_holds_what_the_value_handed_in_holds_there(
    read: Callable[[object], Sequence[ValidationProblem]], value: object
) -> None:
    # The value itself, not a copy a reader made on the way: the problems
    # of a field, a pipeline, a chunk grid over the shape, and a document
    # a group holds alike. A missing key holds none, and nor does what is
    # not JSON, such as a store's bytes.
    problems = read(value)
    assert len(problems) != 0
    for problem in problems:
        there = value_at(value, problem.loc)
        assert problem.input is (there if is_canonical_json(there, finite=False) else UNSET), (
            problem
        )
    if value is BAD_ARRAY:
        assert {problem.loc for problem in problems} >= BAD_ARRAY_LOCS


class WideBits(TypedDict, closed=True):
    bits: Annotated[int, Le(16)]


def test_a_problem_with_what_a_name_carries_is_the_field_s() -> None:
    # `r24` carries `{"bits": 24}`, which this reader's own raw bits refuse:
    # the problem is found at the field, where the name is what is there,
    # and what was expected of `bits` is not expected of the name.
    scope = CORE_AND_EXTENSIONS.extended_with(DataTypeDefinition(name="r*", configuration=WideBits))
    _, problems = resolve("r24", DataTypeDefinition, scope, ("data_type",))
    assert [(p.loc, p.input, dict(p.ctx)) for p in problems] == [(("data_type",), "r24", {})]
    assert problems[0].message == "expected an integer <= 16, got 24"


class Ordered(TypedDict, closed=True):
    order: Literal["F", "C"]


@pytest.mark.parametrize(
    ("problems", "loc", "expected"),
    [
        (check({"order": "Q"}, Ordered)[1], ("order",), ("C", "F")),
        (
            validate_array_metadata_v3({**BAD_ARRAY, "zarr_format": 2}),
            ("zarr_format",),
            (3,),
        ),
        (validate_node_metadata_v3({"node_type": "dataset"}), ("node_type",), ("array", "group")),
        (validate_array_metadata_v2({"order": "Q"}), ("order",), ("C", "F")),
        (
            validate_group_metadata_v3(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "consolidated_metadata": {
                        "kind": "inline",
                        "must_understand": True,
                        "metadata": {},
                    },
                }
            ),
            ("consolidated_metadata", "must_understand"),
            (False,),
        ),
    ],
    ids=["a-literal", "zarr-format", "node-type", "v2-order", "consolidated-must-understand"],
)
def test_a_value_outside_a_closed_set_is_told_the_set(
    problems: Sequence[ValidationProblem], loc: tuple[str | int, ...], expected: tuple[object, ...]
) -> None:
    # In the order the message lists them, as JSON writes them.
    (problem,) = [problem for problem in problems if problem.loc == loc]
    assert dict(problem.ctx) == {"expected": expected}


def test_error_a_ctx_of_another_mapping_type_is_checked() -> None:
    # Only a problem's own `ctx`, checked when it was made, is taken as
    # checked: any other mapping is, whatever its type.
    with pytest.raises(TypeError, match="ctx is an object of JSON values"):
        ValidationProblem(
            (), "m", "invalid_value", ctx=cast("Any", MappingProxyType({"x": object()}))
        )
    first = ValidationProblem((), "m", "invalid_value", ctx={"ge": 0})
    again = dataclasses.replace(first, loc=("a",))
    assert again.ctx == {"ge": 0}
    assert again.ctx is first.ctx


def test_a_problem_at_the_first_element_holds_it() -> None:
    # Checked against the literal, not `value_at`, which the reader uses.
    (problem,) = with_input([ValidationProblem(("a", 0), "bad", "invalid_value")], {"a": [7]})
    assert problem.input == 7
    assert value_at([5, 6], (0,)) == 5
    read = resolve("bytes", DataTypeDefinition, CORE_AND_EXTENSIONS)[0]
    (found,) = fill_value_problems(read, [-1])
    assert (found.loc, found.input) == ((0,), -1)


def test_a_problem_s_ctx_is_its_own_at_every_level() -> None:
    # Copied when the problem is made, at every level, so nothing the
    # caller does to what it handed in reaches the problem.
    inner: dict[str, int] = {"a": 1}
    problem = ValidationProblem((), "m", "invalid_value", ctx={"nested": inner})
    inner["a"] = 2
    assert problem.ctx["nested"] == {"a": 1}


def test_error_a_problem_not_below_where_a_reader_counts_from_is_refused() -> None:
    # `within` relocates a problem from the document handed in to the one
    # read; one located from another root would land in the wrong place.
    problem = ValidationProblem(("a", "b"), "m", "invalid_value")
    assert within((problem,), ("a",))[0].loc == ("b",)
    with pytest.raises(TypeError, match="does not sit below"):
        within((problem,), ("c",))
