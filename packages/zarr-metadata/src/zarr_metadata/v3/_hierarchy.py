"""A Zarr v3 hierarchy: the names and paths of its nodes, and the tree they form.

`NodeName` and `NodePath` are modeled on the zarrs types with the same
names. They enforce the spec's rules, including the reservation of
`zarr.json`. `hierarchy_problems` takes the node type of each node, keyed
by path, and checks that the nodes form a valid tree.

Each function here reports at most one problem per value, with all the
reasons in that one problem. This keeps the size of the report
proportional to the size of the input: a path with a thousand bad names
is one problem, not a thousand problems that each repeat the path.

This module is private. Consumers import `NodeName` and `NodePath` from
`zarr_metadata.v3`, and the validators from `zarr_metadata.model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NewType, TypeGuard, cast

from zarr_metadata._json import MetadataValidationError, ValidationProblem, shown, with_input

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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

NodeType = Literal["array", "group"]
"""The two kinds of node in a hierarchy."""


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
    tree = _Tree()
    for path in well_formed:
        tree.hold(path, nodes[path])
    reported: set[str] = set()
    for path in well_formed:
        if path == "/":
            continue
        holder, missing = tree.above(path)
        if holder == "array":
            # No node may be below an array, no matter how many groups are
            # missing between them.
            message = (
                f"expected a node below a group, got {shown(path)}, below the array "
                f"{shown(tree.holder_of(path))}"
            )
            problems.append(ValidationProblem((path,), message, "invalid_value"))
        elif missing != 0:
            nearest = path.rpartition("/")[0] or "/"
            if nearest not in reported:
                reported.add(nearest)
                above = missing - 1
                more = (
                    "" if above == 0 else f", and {above} group{'s' if above != 1 else ''} above it"
                )
                message = f"missing the group holding {shown(path)}{more}"
                problems.append(ValidationProblem((nearest,), message, "missing_key"))
    return tuple(problems)


class _Branch:
    """A name in the tree of present paths: whether a node is present at it, the node's type, and the names below it."""

    __slots__ = ("children", "held", "node_type")

    def __init__(self) -> None:
        self.held = False
        self.node_type: NodeType | None = None
        self.children: dict[str, _Branch] = {}


class _Tree:
    """The present paths as a tree of their names, so each path is walked once, in time proportional to its length.

    Walking up a path by building each ancestor's string costs the path's
    length for each ancestor. A hostile key of a million characters would
    make that quadratic. Here a path is split once and walked name by name.
    """

    __slots__ = ("root",)

    def __init__(self) -> None:
        self.root = _Branch()

    def hold(self, path: str, node_type: NodeType | None) -> None:
        """Mark the node at `path` (a node path) as present, with type `node_type`."""
        branch = self.root
        for name in _names(path):
            branch = branch.children.setdefault(name, _Branch())
        branch.held = True
        branch.node_type = node_type

    def above(self, path: str) -> tuple[NodeType | None, int]:
        """For the node at `path` (a node path below the root), return the type of the nearest present node above it and the number of groups missing between them. The type is `"group"` for a node of unknown type, and None if no node above is present."""
        names = _names(path)
        branch: _Branch | None = self.root
        holder: NodeType | None = None
        held_depth = -1
        for depth, name in enumerate(names[:-1], start=0):
            if branch is not None and branch.held:
                holder, held_depth = branch.node_type or "group", depth
            branch = None if branch is None else branch.children.get(name)
        # The parent, at the last depth walked, may also be present.
        if branch is not None and branch.held:
            holder, held_depth = branch.node_type or "group", len(names) - 1
        return holder, len(names) - 1 - held_depth

    def holder_of(self, path: str) -> str:
        """The path of the nearest present node above the node at `path`. `path` must be a node path below the root, with a present node above it."""
        names = _names(path)
        branch: _Branch | None = self.root
        holder = "/"
        for depth, name in enumerate(names[:-1]):
            if branch is not None and branch.held:
                holder = "/" + "/".join(names[:depth]) if depth != 0 else "/"
            branch = None if branch is None else branch.children.get(name)
        if branch is not None and branch.held:
            holder = "/" + "/".join(names[:-1])
        return holder


def _names(path: str) -> list[str]:
    """The names between a node path's `/` characters. Empty for the root's path."""
    return [] if path == "/" else path[1:].split("/")


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
    "NodeType",
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
