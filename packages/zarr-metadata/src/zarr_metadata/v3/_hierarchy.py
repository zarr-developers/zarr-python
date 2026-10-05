"""A Zarr v3 hierarchy: the names and paths of its nodes, and the tree they make.

`NodeName` and `NodePath` are modeled on zarrs' types of those names, and
hold the spec's rules, which reserve `zarr.json` too. `hierarchy_problems`
judges the node type of each node of a hierarchy, by its path, as a tree.

Every problem here is one per value, and says its reasons at once, so
what is reported stays proportional to what was read: a path of a
thousand bad names is one problem, not a thousand each repeating the
path.

Private: consumers import `NodeName` and `NodePath` from `zarr_metadata.v3`,
and the validators from `zarr_metadata.model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NewType, TypeGuard, cast

from zarr_metadata._json import MetadataValidationError, ValidationProblem, shown, with_input

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

NodeName = NewType("NodeName", str)
"""The name of a node in a Zarr v3 hierarchy.

The root's is `""`. Any other is not empty, holds no `/`, is not periods
alone -- `.`, `..` -- does not start with the reserved `__`, and is not
`zarr.json`. Case matters: `foo` and `FOO` are two names
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L818-L837).
"""

NodePath = NewType("NodePath", str)
"""The path of a node in a Zarr v3 hierarchy.

The root's is `/`. Any other's is its parent's path, a `/` unless the
parent is the root, and its name: so a path starts with `/`, does not end
with one, and holds a node name between each two
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L211-L229).
"""

NodeType = Literal["array", "group"]
"""The two kinds of node a hierarchy holds."""


def said(faults: Sequence[str]) -> str:
    """`faults`, each a reason, as one clause: `a`, `a and b`, `a, b and c`."""
    if len(faults) <= 1:
        return "".join(faults)
    return f"{', '.join(faults[:-1])} and {faults[-1]}"


def name_faults(name: str) -> list[str]:
    """What keeps `name` from being a node name, each said; none for a node name, the root's `""` among them."""
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
    """What keeps `names`, the names a path holds between its `/`, from being node names below the root: the first that is not one, said, and how many more there are."""
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
    """What keeps `path` from being a node path, each said; none for a node path, the root's `/` among them."""
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
    """What keeps `nodes`, the node type of each node by its path, from being a Zarr v3 hierarchy: one problem per node it is about, at that node's path.

    "A Zarr hierarchy is a tree structure, where each node in the tree is
    either a group or an array. Group nodes may have children but array
    nodes may not"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L177-L181).
    So each path is a node path, and each node but the root has its parent
    among them, a group: a node below an array is an `invalid_value` at its
    path, and the nearest group missing above a node is a `missing_key` at
    that group's path, once, saying how many more are missing above it, the
    root's among them. A node of no type known, None, is taken as a group:
    what makes its type unknown is its own problem, reported where it sits.
    A mapping holds each path once, so no two siblings share a name.
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
            # No node is below an array, however many groups are missing
            # between them.
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
    """A name in the tree of held paths: whether a node is held there, its type, and the names below it."""

    __slots__ = ("children", "held", "node_type")

    def __init__(self) -> None:
        self.held = False
        self.node_type: NodeType | None = None
        self.children: dict[str, _Branch] = {}


class _Tree:
    """The held paths as a tree of their names, so each path is walked once, in time proportional to its length.

    Walking up a path by its ancestors' strings costs the path's length
    for each ancestor, which a hostile key of a million characters makes
    a quadratic wait; here a path is split once and walked name by name.
    """

    __slots__ = ("root",)

    def __init__(self) -> None:
        self.root = _Branch()

    def hold(self, path: str, node_type: NodeType | None) -> None:
        """Mark the node at `path`, a node path, as held, of `node_type`."""
        branch = self.root
        for name in _names(path):
            branch = branch.children.setdefault(name, _Branch())
        branch.held = True
        branch.node_type = node_type

    def above(self, path: str) -> tuple[NodeType | None, int]:
        """Of the node at `path`, a node path below the root: the type of the nearest held node above it, `"group"` for one of no type known and None when none is held, and how many groups are missing between them."""
        names = _names(path)
        branch: _Branch | None = self.root
        holder: NodeType | None = None
        held_depth = -1
        for depth, name in enumerate(names[:-1], start=0):
            if branch is not None and branch.held:
                holder, held_depth = branch.node_type or "group", depth
            branch = None if branch is None else branch.children.get(name)
        # The parent, at the last depth walked, may be held too.
        if branch is not None and branch.held:
            holder, held_depth = branch.node_type or "group", len(names) - 1
        return holder, len(names) - 1 - held_depth

    def holder_of(self, path: str) -> str:
        """The path of the nearest held node above the node at `path`, a node path below the root that has one."""
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
    """The names a node path holds between its `/`: none for the root's."""
    return [] if path == "/" else path[1:].split("/")


def _string_problems(value: object, what: str) -> tuple[ValidationProblem, ...]:
    return (ValidationProblem((), f"expected {what}, got {shown(value)}", "invalid_type"),)


def validate_node_name_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a v3 node name, said in one `invalid_value`, or an `invalid_type` for what is not a string."""
    if not isinstance(value, str):
        return with_input(_string_problems(value, "a node name"), value)
    faults = name_faults(value)
    if len(faults) == 0:
        return ()
    message = f"expected a node name, got {shown(value)}, which {said(faults)}"
    return with_input((ValidationProblem((), message, "invalid_value"),), value)


def is_node_name_v3(value: object) -> TypeGuard[NodeName]:
    """Whether `value` is a v3 node name `validate_node_name_v3` finds nothing wrong with."""
    return len(validate_node_name_v3(value)) == 0


def parse_node_name_v3(value: object) -> NodeName:
    """`value` as a `NodeName`, or `MetadataValidationError` with every reason it is not one."""
    problems = validate_node_name_v3(value)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return NodeName(cast("str", value))


def validate_node_path_v3(value: object) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a v3 node path, said in one `invalid_value`, or an `invalid_type` for what is not a string."""
    if not isinstance(value, str):
        return with_input(_string_problems(value, "a node path"), value)
    faults = path_faults(value)
    if len(faults) == 0:
        return ()
    message = f"expected a node path, got {shown(value)}, which {said(faults)}"
    return with_input((ValidationProblem((), message, "invalid_value"),), value)


def is_node_path_v3(value: object) -> TypeGuard[NodePath]:
    """Whether `value` is a v3 node path `validate_node_path_v3` finds nothing wrong with."""
    return len(validate_node_path_v3(value)) == 0


def parse_node_path_v3(value: object) -> NodePath:
    """`value` as a `NodePath`, or `MetadataValidationError` with every reason it is not one."""
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
