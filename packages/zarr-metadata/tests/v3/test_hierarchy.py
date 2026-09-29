"""A Zarr v3 hierarchy: the names and paths of its nodes, and the tree they make, as the core specification constrains them.

Names:
https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L818-L837
Paths:
https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L211-L229
The tree:
https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L177-L181
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, cast

import pytest

from zarr_metadata.model import (
    MetadataValidationError,
    ValidationProblem,
    is_node_name_v3,
    is_node_path_v3,
    parse_node_name_v3,
    parse_node_path_v3,
    validate_node_name_v3,
    validate_node_path_v3,
)
from zarr_metadata.v3._hierarchy import hierarchy_problems

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("value", "message"),
    [
        # The root's name, and names of any code points but the reserved.
        ("", None),
        ("a", None),
        ("FOO", None),
        ("a.b", None),
        ("...a", None),
        ("_a", None),
        ("a__", None),
        ("zarr.json.bak", None),
        ("ñ", None),
        ("/", 'expected a node name, got "/", which holds "/"'),
        ("a/b", 'expected a node name, got "a/b", which holds "/"'),
        (".", 'expected a node name, got ".", which is periods alone'),
        ("..", 'expected a node name, got "..", which is periods alone'),
        ("__a", 'expected a node name, got "__a", which starts with the reserved "__"'),
        ("zarr.json", 'expected a node name, got "zarr.json", which is the reserved "zarr.json"'),
        # Every reason a name is not one, in one problem.
        (
            "__/",
            'expected a node name, got "__/", which holds "/" and starts with the reserved "__"',
        ),
    ],
)
def test_node_names(value: str, message: str | None) -> None:
    problems = validate_node_name_v3(value)
    assert [problem.message for problem in problems] == ([] if message is None else [message])
    assert {(problem.loc, problem.kind, problem.input) for problem in problems} <= {
        ((), "invalid_value", value)
    }
    assert is_node_name_v3(value) is (message is None)
    if message is None:
        assert parse_node_name_v3(value) == value


@pytest.mark.parametrize(
    ("value", "message"),
    [
        # The root's path, and the paths below it.
        ("/", None),
        ("/a", None),
        ("/a/b", None),
        ("/a/.b/c..", None),
        ("", 'expected a node path, got "", which does not start with "/"'),
        ("a/b", 'expected a node path, got "a/b", which does not start with "/"'),
        ("/a/", 'expected a node path, got "/a/", which ends with "/"'),
        (
            "//",
            'expected a node path, got "//", which ends with "/" and holds an empty name between two "/"',
        ),
        ("/a//b", 'expected a node path, got "/a//b", which holds an empty name between two "/"'),
        (
            "/a/../b",
            'expected a node path, got "/a/../b", which holds "..", a name that is periods alone',
        ),
        (
            "/__a",
            (
                'expected a node path, got "/__a", which holds "__a", a name that starts with the '
                'reserved "__"'
            ),
        ),
        (
            "/a/zarr.json",
            (
                'expected a node path, got "/a/zarr.json", which holds "zarr.json", a name that is '
                'the reserved "zarr.json"'
            ),
        ),
        # Every reason a path is not one, in one problem: the first name that
        # is not a node name is said, and the rest counted.
        (
            "/./__a/",
            (
                'expected a node path, got "/./__a/", which ends with "/", holds ".", a name that is '
                "periods alone and holds 1 more name that is not a node name"
            ),
        ),
    ],
)
def test_node_paths(value: str, message: str | None) -> None:
    problems = validate_node_path_v3(value)
    assert [problem.message for problem in problems] == ([] if message is None else [message])
    assert {(problem.loc, problem.kind, problem.input) for problem in problems} <= {
        ((), "invalid_value", value)
    }
    assert is_node_path_v3(value) is (message is None)
    if message is None:
        assert parse_node_path_v3(value) == value


def test_error_parse_node_name_raises_every_reason_a_string_is_not_a_node_name() -> None:
    with pytest.raises(MetadataValidationError) as raised:
        parse_node_name_v3("__/")
    assert raised.value.problems == validate_node_name_v3("__/")
    assert len(raised.value.problems) == 1
    assert 'holds "/"' in raised.value.problems[0].message
    assert 'starts with the reserved "__"' in raised.value.problems[0].message


def test_error_parse_node_path_raises_every_reason_a_string_is_not_a_node_path() -> None:
    with pytest.raises(MetadataValidationError) as raised:
        parse_node_path_v3("/./__a/")
    assert raised.value.problems == validate_node_path_v3("/./__a/")
    assert len(raised.value.problems) == 1
    assert 'ends with "/"' in raised.value.problems[0].message
    assert "1 more name" in raised.value.problems[0].message


@pytest.mark.parametrize(
    ("validate", "parse", "what"),
    [
        (validate_node_name_v3, parse_node_name_v3, "a node name"),
        (validate_node_path_v3, parse_node_path_v3, "a node path"),
    ],
    ids=["name", "path"],
)
def test_error_a_node_name_or_path_is_a_string(
    validate: Callable[[object], tuple[ValidationProblem, ...]],
    parse: Callable[[object], str],
    what: str,
) -> None:
    problems = validate(3)
    assert problems == (ValidationProblem((), f"expected {what}, got 3", "invalid_type"),)
    assert problems[0].input == 3
    with pytest.raises(MetadataValidationError) as raised:
        parse(b"a")
    assert [(problem.loc, problem.kind) for problem in raised.value.problems] == [
        ((), "invalid_type")
    ]


@pytest.mark.parametrize(
    "nodes",
    [
        # No nodes, a root alone, and a tree of groups whose leaves are
        # arrays; the root may be an array, which is then the hierarchy.
        {},
        {"/": "group"},
        {"/": "array"},
        {"/": "group", "/a": "array", "/g": "group", "/g/b": "array", "/g/h": "group"},
        # A node of no type known is taken as a group.
        {"/": "group", "/x": None, "/x/a": "array"},
    ],
    ids=["empty", "root-group", "root-array", "tree", "unknown-type"],
)
def test_a_hierarchy(nodes: dict[str, Literal["array", "group"] | None]) -> None:
    assert hierarchy_problems(nodes) == ()


def test_error_a_hierarchy_holds_its_nodes_at_node_paths() -> None:
    assert hierarchy_problems({"/": "group", "a": "array", "/b/": "array"}) == (
        ValidationProblem(
            ("a",), 'expected a node path, got "a", which does not start with "/"', "invalid_value"
        ),
        ValidationProblem(
            ("/b/",), 'expected a node path, got "/b/", which ends with "/"', "invalid_value"
        ),
    )


@pytest.mark.parametrize(
    ("nodes", "path"),
    [
        ({"/": "group", "/a": "array", "/a/b": "group"}, "/a/b"),
        # However many groups are missing between them: none would help.
        ({"/": "group", "/a": "array", "/a/b/c": "array"}, "/a/b/c"),
        # The root is an array: the hierarchy is that array alone.
        ({"/": "array", "/a": "group"}, "/a"),
    ],
    ids=["child", "descendant", "below-the-root"],
)
def test_error_no_node_is_below_an_array(
    nodes: dict[str, Literal["array", "group"] | None], path: str
) -> None:
    holder = next(ancestor for ancestor in ("/a", "/") if nodes.get(ancestor) == "array")
    message = f'expected a node below a group, got "{path}", below the array "{holder}"'
    assert hierarchy_problems(nodes) == (ValidationProblem((path,), message, "invalid_value"),)


def test_error_a_hierarchy_holds_the_group_holding_each_node() -> None:
    # The nearest group missing above a node, once, counting those above it:
    # the root's among them.
    assert hierarchy_problems({"/a/b/c": "array", "/a/b/d": "array"}) == (
        ValidationProblem(
            ("/a/b",), 'missing the group holding "/a/b/c", and 2 groups above it', "missing_key"
        ),
    )
    assert hierarchy_problems({"/": "group", "/a/b": "array"}) == (
        ValidationProblem(("/a",), 'missing the group holding "/a/b"', "missing_key"),
    )


def test_problems_stay_proportional_to_the_path() -> None:
    # One problem per value, however many names are wrong and however many
    # groups are missing, so a hostile path costs what it weighs.
    path = "/" * 10_000
    problems = validate_node_path_v3(path)
    assert len(problems) == 1
    assert len(problems[0].message) < len(path) + 200
    deep = "/a" * 5_000
    found = hierarchy_problems({deep: "array"})
    assert [(problem.loc, problem.kind) for problem in found] == [
        ((deep.rpartition("/")[0],), "missing_key")
    ]
    assert found[0].message.endswith(", and 4999 groups above it")
    assert len(found[0].message) < len(deep) + 200


def test_a_hierarchy_is_walked_in_time_proportional_to_its_paths() -> None:
    # Each path split once and walked name by name: a key of a million
    # characters, and thousands of keys under one long missing prefix,
    # each cost what they weigh, where walking up by ancestor strings
    # costs the square.
    started = time.perf_counter()
    long = "/a" * 500_000
    assert len(hierarchy_problems({long: "array"})) == 1
    prefix = "/a" * 5_000
    many = {f"{prefix}/{index}": "array" for index in range(2_000)}
    assert len(hierarchy_problems(cast("dict[str, Literal['array', 'group'] | None]", many))) == 1
    assert time.perf_counter() - started < 10
