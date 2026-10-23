"""A Zarr v3 hierarchy: the names and paths of its nodes, and the tree they form.

`NodeName` and `NodePath` are modeled on the zarrs types with the same
names. They enforce the spec's rules, including the reservation of
`zarr.json`. `hierarchy_problems` takes the node type of each node, keyed
by path, checks each path against the v3 rules, and then checks that the
nodes form a valid tree with the shared `tree_problems`.

Each function here reports at most one problem per value, with all the
reasons in that one problem. This keeps the size of the report
proportional to the size of the input: a path with a thousand bad names
is one problem, not a thousand problems that each repeat the path.

This module is private. Consumers import `NodeName` and `NodePath` from
`zarr_metadata.v3`, and the validators from `zarr_metadata.model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NewType, TypeGuard, cast

from zarr_metadata._hierarchy import tree_problems
from zarr_metadata._json import MetadataValidationError, ValidationProblem, shown, with_input

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr_metadata._hierarchy import NodeType

NodeName = NewType("NodeName", str)
"""The name of a node in a Zarr v3 hierarchy.

The root's name is `""`. Any other name is not empty, contains no `/`, is
not made only of periods (such as `.` or `..`), does not start with the
reserved `__`, and is not `zarr.json`. Names are case-sensitive: `foo` and
`FOO` are two different names
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L818-L837).
"""

NodePath = NewType("NodePath", str)
"""The path of a node in a Zarr v3 hierarchy.

The root's path is `/`. Any other node's path is its parent's path,
followed by `/` (unless the parent is the root), followed by its name. So
a path starts with `/`, does not end with `/`, and has a node name between
each pair of `/`
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L211-L229).
"""


def said(faults: Sequence[str]) -> str:
    """Join `faults`, each a reason, into one clause: `a`, `a and b`, `a, b and c`."""
    if len(faults) <= 1:
        return "".join(faults)
    return f"{', '.join(faults[:-1])} and {faults[-1]}"


def name_faults(name: str) -> list[str]:
    """The reasons `name` is not a node name, each as a message. Empty for a valid node name, including the root's `""`."""
    faults: list[str] = []
    if "/" in name:
        faults.append('holds "/"')
    if name != "" and set(name) == {"."}:
        faults.append("is periods alone")
    if name.startswith("__"):
        faults.append('starts with the reserved "__"')
    if name == "zarr.json":
        faults.append('is the reserved "zarr.json"')
    return faults


def names_faults(names: list[str]) -> list[str]:
    """The reasons `names` (the names between a path's `/` characters) are not all valid node names below the root. Reports the first invalid name and how many more invalid names there are."""
    faulty = [name for name in names if name == "" or len(name_faults(name)) != 0]
    if len(faulty) == 0:
        return []
    first = faulty[0]
    faults = (
        ['holds an empty name between two "/"']
        if first == ""
        else [f"holds {shown(first)}, a name that {said(name_faults(first))}"]
    )
    if len(faulty) > 1:
        more = len(faulty) - 1
        faults.append(
            f"holds {more} more name{'s' if more != 1 else ''} that {'are' if more != 1 else 'is'} not a node name"
        )
    return faults


def path_faults(path: str) -> list[str]:
    """The reasons `path` is not a node path, each as a message. Empty for a valid node path, including the root's `/`."""
    if path == "/":
        return []
    if not path.startswith("/"):
        return ['does not start with "/"']
    faults: list[str] = []
    names = path[1:].split("/")
    if path.endswith("/"):
        faults.append('ends with "/"')
        names = names[:-1]
    return [*faults, *names_faults(names)]


def hierarchy_problems(nodes: Mapping[str, NodeType | None]) -> tuple[ValidationProblem, ...]:
    """The reasons `nodes` (the node type of each node, keyed by path) is not a Zarr v3 hierarchy. Reports one problem per affected node, at that node's path.

    "A Zarr hierarchy is a tree structure, where each node in the tree is
    either a group or an array. Group nodes may have children but array
    nodes may not"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L177-L181).
    So each path must be a node path, and each node except the root must
    have its parent in `nodes`, and that parent must be a group. A node
    below an array is an `invalid_value` at the node's path. If groups are
    missing above a node, the nearest missing group is reported once as a
    `missing_key` at that group's path. The message says how many more
    groups are missing above it, including the root. A node whose type is
    unknown (None) is treated as a group. The cause of the unknown type is
    a separate problem, reported at that node's location. A mapping
    contains each path only once, so no two siblings can share a name.
    """
    problems: list[ValidationProblem] = []
    well_formed: list[str] = []
    for path in nodes:
        faults = path_faults(path)
        if len(faults) == 0:
            well_formed.append(path)
        else:
            message = f"expected a node path, got {shown(path)}, which {said(faults)}"
            problems.append(ValidationProblem((path,), message, "invalid_value"))
    problems.extend(tree_problems({path: nodes[path] for path in well_formed}))
    return tuple(problems)


def _string_problems(value: object, what: str) -> tuple[ValidationProblem, ...]:
    return (ValidationProblem((), f"expected {what}, got {shown(value)}", "invalid_type"),)


def validate_node_name_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a v3 node name, reported in one `invalid_value` problem, or an `invalid_type` problem if `value` is not a string."""
    if not isinstance(value, str):
        return with_input(_string_problems(value, "a node name"), value)
    faults = name_faults(value)
    if len(faults) == 0:
        return ()
    message = f"expected a node name, got {shown(value)}, which {said(faults)}"
    return with_input((ValidationProblem((), message, "invalid_value"),), value)


def is_node_name_v3(value: object) -> TypeGuard[NodeName]:
    """Whether `value` is a v3 node name, meaning `validate_node_name_v3` finds no problems."""
    return len(validate_node_name_v3(value)) == 0


def parse_node_name_v3(value: object) -> NodeName:
    """Return `value` as a `NodeName`, or raise `MetadataValidationError` with every reason it is not one."""
    problems = validate_node_name_v3(value)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return NodeName(cast("str", value))


def validate_node_path_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a v3 node path, reported in one `invalid_value` problem, or an `invalid_type` problem if `value` is not a string."""
    if not isinstance(value, str):
        return with_input(_string_problems(value, "a node path"), value)
    faults = path_faults(value)
    if len(faults) == 0:
        return ()
    message = f"expected a node path, got {shown(value)}, which {said(faults)}"
    return with_input((ValidationProblem((), message, "invalid_value"),), value)


def is_node_path_v3(value: object) -> TypeGuard[NodePath]:
    """Whether `value` is a v3 node path, meaning `validate_node_path_v3` finds no problems."""
    return len(validate_node_path_v3(value)) == 0


def parse_node_path_v3(value: object) -> NodePath:
    """Return `value` as a `NodePath`, or raise `MetadataValidationError` with every reason it is not one."""
    problems = validate_node_path_v3(value)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return NodePath(cast("str", value))


__all__ = [
    "NodeName",
    "NodePath",
    "hierarchy_problems",
    "is_node_name_v3",
    "is_node_path_v3",
    "name_faults",
    "names_faults",
    "parse_node_name_v3",
    "parse_node_path_v3",
    "path_faults",
    "said",
    "validate_node_name_v3",
    "validate_node_path_v3",
]
